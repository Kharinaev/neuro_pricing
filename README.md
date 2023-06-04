# Neuro pricing
Final qualifying paper of the CMC MSU bachelor's degree

# Neural network for options pricing and its comparison with classical methods / Нейросеть для ценообразования опционов и ее сравнение с классическими методами

## Work summary
- considered options pricing methods that modeling the [volatility smile](https://en.wikipedia.org/wiki/Volatility_smile)
- [SABR](https://www.researchgate.net/publication/235622441_Managing_Smile_Risk), [Heston](https://faculty.baruch.cuny.edu/lwu/890/Heston93.pdf) and [Variance-Gamma](https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf) models was implemented and applied on real data from [yfinance](https://pypi.org/project/yfinance/)
- neural networks was proposed and implemented for modeling the volatility smile
- achieved comparable quality of restoring the volatility surface with classical methods with the following advantages of neural networks:
  - no need to retrain neural networks when changing options parameters
  - the possibility of fast calculations on graphics processors (GPU)

## Краткое описание работы
- рассмотрены методы ценообразования опционов, моделирующие [улыбку волатильности](https://en.wikipedia.org/wiki/Volatility_smile)
- реализованы и применены модели [SABR](https://www.researchgate.net/publication/235622441_Managing_Smile_Risk), [Heston](https://faculty.baruch.cuny.edu/lwu/890/Heston93.pdf) и [Variance-Gamma](https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf) на реальных данных из [yfinance](https://pypi.org/project/yfinance/)
- были предложены и применены нейронные сети для моделирования улыбки волатильности
- достигнуто сравнимое качество восстановления поверхности волатильности с классическими методами при следующих преимуществах нейронных сетей:
  - нет необходимости переобучения нейронных сетей при изменении параметров опционов
  - возможность быстрых вычислений на графических процессорах (GPU)


## Volatility surface example / Пример поверхности волатильности
