import fftoptionlib as fft
import QuantLib as ql
import numpy as np
from scipy.optimize import minimize

class SABR:
    
    def __init__(self, stock_price):
        self.stock_price = stock_price
        self.params = []
    
    @staticmethod
    def volatility(strikes, stock_price, t, alpha, beta, rho, nu):
        f = stock_price
        K = strikes
        z = nu / alpha * (f*K)**((1-beta)/2) * np.log(f/K)
        x = np.log(((1-2*rho*z+z**2)**0.5 + z - rho)/(1-rho))
        first = alpha / ((f*K)**((1-beta)/2) * (1+(1-beta)**2 / 24 * 
                (np.log(f/K))**2 + (1-beta)**4 / 1920 * (np.log(f/K))**4))
        second = z/x
        third = 1 + t*((1-beta)**2 / 24 *  alpha**2 / ((f*K)**(1-beta)) + 
                (rho*beta*nu*alpha)/(4*(f*K)**((1-beta)/2)) + nu**2*(2-3*rho**2)/24)
        return first * second * third

    def fit(self, strikes, mrkt_vols, expiration:float, frac_ootm=1, frac_itm=1):
        
        def RSS_SABR(params, t, stock_price):
            sabr_vols = self.volatility(strikes, stock_price, t, *params)
            return np.sum(np.square(sabr_vols - mrkt_vols))
        
        strikes = np.array(strikes)
        mrkt_vols = np.array(mrkt_vols)

        if frac_ootm < 1 and frac_itm < 1:
            nearest = np.argmin(np.abs(strikes-self.stock_price))
            
            mask_relevant = np.repeat(False, len(strikes))
            rel_ootm = int(frac_ootm * len(strikes) / 2)
            rel_itm = int(frac_itm * len(strikes) / 2)
            mask_relevant[nearest-rel_itm : nearest+rel_ootm+1] = True
            
            strikes = strikes[mask_relevant]
            mrkt_vols = mrkt_vols[mask_relevant]

        # alpha, beta, rho, nu
        min_args = (expiration, self.stock_price)
        x0 = [0.001, 1, -0.999, 0.001]
        bounds = ((0.001, None), (1, 1), (-0.999, 0.999), (0.001, None))
        res = minimize(RSS_SABR, x0, min_args, bounds=bounds)
        self.params.append(res.x)

    def iv_surface(self, strikes, expirations, params=None):
        if params is None:
            params = self.params
        surface = np.zeros((len(expirations), len(strikes)), dtype=float)
        for i, (t,p) in enumerate(zip(expirations, params)):
            surface[i,:] = self.volatility(strikes, self.stock_price, t, *p) 
        
        return surface

    def MSE_all(self, df_list, expirations, params=None):
        if params is None:
            params = self.params
        
        sse = 0
        cnt = 0
        for i, (t,df,p) in enumerate(zip(expirations, df_list, params)):
            cnt += len(df)
            sabr_vols = self.volatility(df.strike, self.stock_price, t, *p)
            sse += np.sum(np.square(sabr_vols - df.impliedVolatility))

        return sse / cnt
    

class Heston(ql.HestonModel):
    
    def __init__(
        self, 
        spot,
        init_params=(0.01, 1, 0.5, 0, 0.1), 
        bounds=[(0,1), (0.01,15), (0.01,None), (-1,1), (0,1.0) ],
        q=0.0, 
        r=0.05, 
        evaluation_date='2022-03-16'
    ):
        self.init_params = init_params
        self.bounds = bounds
        theta, kappa, sigma, rho, v0 = self.init_params
        self.spot = spot
        self.calculation_date = ql.Date(evaluation_date, '%Y-%m-%d')
        day_count = ql.Actual365Fixed() 
        self.calendar = ql.UnitedStates(1)
        self.yield_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.calculation_date, r, day_count)
        )
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.calculation_date, q, day_count)
        )
        process = ql.HestonProcess(
            self.yield_ts, self.dividend_ts, 
            ql.QuoteHandle(ql.SimpleQuote(spot)), 
            v0, kappa, theta, sigma, rho
        )
        super().__init__(process)
        self.engine = ql.AnalyticHestonEngine(self)
        self.vol_surf = ql.HestonBlackVolSurface(ql.HestonModelHandle(self), self.engine.Gatheral)
        

    def build_helpers(self, expirations, strikes, vols):
        maturities = [
            ql.Period(ql.Date(expir, '%Y-%m-%d') - self.calculation_date, ql.Days)
             for expir in expirations
        ]
        temp=[]
        for m,v in zip(maturities,vols):
            for i,s in enumerate(strikes):
                temp.append( 
                    ql.HestonModelHelper(
                        m, self.calendar, self.spot, s, 
                        ql.QuoteHandle(ql.SimpleQuote(v[i])), 
                        self.yield_ts, self.dividend_ts
                    )  
                )
        for x in temp: x.setPricingEngine(self.engine)
        self.helpers=temp
        self.loss= [x.calibrationError() for x in self.helpers]


    def f_cost(self, params, strikes, mrkt_vols, expirations, norm=False):
        self.setParams( ql.Array(list(params)) )
        self.build_helpers(expirations, strikes, mrkt_vols)
        if norm == True:
            loss = np.array(self.loss)
            mask = np.isinf(loss) | np.isnan(loss)
            self.loss = np.mean(np.square(loss[~mask]))
        return self.loss


    def fit(self, strikes, mrkt_vols, expirations, method='L-BFGS-B'):
        # self.build_heplers(expirations, strikes, mrkt_vols)
        if method == 'L-BFGS-B':
            min_args = (strikes, mrkt_vols, expirations, True)
            res = minimize(
                self.f_cost, self.init_params, 
                args=min_args, 
                method='L-BFGS-B', 
                bounds=self.bounds
            )
            self.params = res.x
        elif method == 'LM':
            self.build_helpers(expirations, strikes, mrkt_vols)
            self.calibrate(
                self.helpers, 
                ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8),
                ql.EndCriteria(500, 300, 1.0e-8,1.0e-8, 1.0e-8)
            )


    def iv_surface(self, strikes, maturities):
        surface = np.zeros((len(maturities), len(strikes)), dtype=float)
        for i,t in enumerate(maturities):
            surface[i,:] = np.array([self.vol_surf.blackVol(t,s) for s in strikes])
        
        return surface

    def MSE_all(self, df_list, maturities):
        sse = 0
        cnt = 0
        for i, (t, df) in enumerate(zip(maturities, df_list)):
            cnt += len(df)
            heston_vols = np.array([self.vol_surf.blackVol(t,s) for s in df.strike])
            sse += np.sum(np.square(heston_vols - df.impliedVolatility))

        return sse / cnt
    

class Variance_Gamma:
    
    def __init__(self, stock_price, evaluation_date='2022-03-16', q=0.0, r=0.05):
        self.evaluation_date = evaluation_date
        ql.Settings.instance().evaluationDate = ql.Date(evaluation_date, '%Y-%m-%d')
        self.params = []
        self.stock_price = stock_price
        self.q = q
        self.r = r

    def get_option_obj(
        self,
        expiration_date : str,
    ):  
        """Build `fftoptionlib` Option object 

        Args:
            expiration_date (str): expiration date in format '%Y-%m-%d'
            q (float): dividend rate
            r (float): risk-free rate

        Returns:
            vanilla_option : `fftoptionlib` Option object
        """
        vanilla_option = fft.BasicOption()
        (vanilla_option.set_underlying_close_price(self.stock_price)
            .set_dividend(self.q)
            .set_maturity_date(expiration_date)
            .set_evaluation_date(self.evaluation_date)
            .set_zero_rate(self.r))

        return vanilla_option


    def get_vg_prices(
        self,
        theta : float,
        v : float,
        sigma : float,
        option_obj,
        strikes, 
    ):
        """Calculates fair option prices using Variance-Gamma model 

        Args:
            theta (float): VG model parameter - drift
            v (float): VG model parameter - variance rate
            sigma (float): VG model parameter - instantaneous volatility
            option_obj : `fftoptionlib` Option object
            strikes : strikes for which needs to calculate prices

        Returns:
            vg_prices : VG-calculated fair option prices
        """
        N=2**15
        d_u = 0.01
        alpha = 1
        ft_pricer = fft.FourierPricer(option_obj)
        (ft_pricer.set_pricing_engine(fft.FFTEngine(N, d_u, alpha, spline_order=2))
            .set_log_st_process(fft.VarianceGamma(theta=theta, v=v, sigma=sigma))
        )

        strikes = np.array(strikes)
        put_call = 'call'

        return ft_pricer.calc_price(strikes, put_call)


    def get_iv_params(
        self,
        expiration_date, 
        strikes,
        sigma_0=0.2,
    ):
        calendar = ql.UnitedStates(1)
        exercise = ql.EuropeanExercise(ql.Date(expiration_date, '%Y-%m-%d'))
        
        S = ql.QuoteHandle(ql.SimpleQuote(self.stock_price))
        q = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, self.q, ql.Actual365Fixed()))
        r = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, self.r, ql.Actual365Fixed()))
        sigma = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(0, calendar, sigma_0, ql.Actual365Fixed())
        )

        return exercise, S, q, r, sigma

    def calc_iv(
        self,
        opt_prices, 
        strikes, 
        iv_params,
    ):
        """Calculates implied volatilities from option prices

        Args:
            iv_params (tuple) : exercise, S, q, r, sigma
        """
        ivs = []
        for V, K in zip(opt_prices, strikes):
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
            option = ql.EuropeanOption(payoff, iv_params[0])
            process = ql.BlackScholesMertonProcess(*iv_params[1:])
            ivs.append(option.impliedVolatility(V, process))

        return np.array(ivs)


    def fit(
        self,
        expiration_date : str,
        strikes,
        mrkt_vols,
        frac_ootm : float=0.6,
        frac_itm : float=0.6,
    ):

        def RSS(par, mkrt_vols, option_obj, strikes, iv_params):
            theta = par[0]
            v = par[1]
            sigma = par[2]
            vg_prices = self.get_vg_prices(theta, v, sigma, option_obj, strikes)
            vg_pred_iv = self.calc_iv(vg_prices, strikes, iv_params)

            return np.sum(np.square(mkrt_vols - vg_pred_iv))

        strikes = np.array(strikes)
        mrkt_vols = np.array(mrkt_vols)
        
        option_obj = self.get_option_obj(expiration_date)
        iv_params = self.get_iv_params(expiration_date, strikes)
        
        if frac_ootm < 1 and frac_itm < 1:
            nearest = np.argmin(np.abs(strikes-self.stock_price))
            mask_relevant = np.repeat(False, len(strikes))
            rel_ootm = int(frac_ootm * len(strikes) / 2)
            rel_itm = int(frac_itm * len(strikes) / 2)
            mask_relevant[nearest-rel_itm : nearest+rel_ootm+1] = True
        else:
            mask_relevant = np.repeat(True, len(strikes))

        min_args = (mrkt_vols[mask_relevant], option_obj, strikes[mask_relevant], iv_params)
        # theta, v, sigma
        x0 = [0, 0.2, 0.3]
        bounds = ((-1,None), (1e-3,None), (1e-3,None))
        res = minimize(RSS, x0, min_args, bounds=bounds, method='L-BFGS-B')
        self.params.append(res.x)

    def iv_surface(self, strikes, expirations, params=None):
        if params is None:
            params = self.params

        surface = np.zeros((len(expirations), len(strikes)), dtype=float)
        for i, (expir,p) in enumerate(zip(expirations, params)):
            option_obj = self.get_option_obj(expir)
            iv_params = self.get_iv_params(expir, strikes)
            vg_prices = self.get_vg_prices(*p, option_obj, strikes)
            surface[i,:] = self.calc_iv(vg_prices, strikes, iv_params)
        
        return surface

    def MSE_all(self, df_list, expirations, params=None):
        if params is None:
            params = self.params

        sse = 0
        cnt = 0
        for i, (expir, df, p) in enumerate(zip(expirations, df_list, params)):
            cnt += len(df)

            option_obj = self.get_option_obj(expir)
            iv_params = self.get_iv_params(expir, df.strike)
            vg_prices = self.get_vg_prices(*p, option_obj, df.strike)
            vg_vols = self.calc_iv(vg_prices, df.strike, iv_params)

            sse += np.sum(np.square(vg_vols - df.impliedVolatility))

        return sse / cnt