# ConvRFF - Implementación de modelos con Random Fourier Features in Convolutional Form

Este repositorio contiene la implementación de modelos de segmentación utilizando la técnica Random Fourier Features in Convolutional Form (ConvRFF), que incluye arquitecturas como Unet, ResUnet y FCN.

## Descripción

La técnica ConvRFF es una forma eficiente de aproximar las operaciones de convolución mediante la proyección de los filtros convolucionales en el espacio de características de Fourier aleatorias. Esto permite acelerar el cálculo de convoluciones en redes neuronales convolucionales (CNN) y reduce significativamente el costo computacional en comparación con el uso de convoluciones tradicionales.

## Contenido

El contenido de este repositorio se organiza de la siguiente manera:

- `models/`: Carpeta que contiene las implementaciones de modelos de segmentación utilizando ConvRFF.
- `models/layers`: Carpeta que incluye las funciones y clases necesarias para la utilización de la técnica ConvRFF.

## Licencia

Este proyecto está bajo la Licencia MIT license. Consulte el archivo [LICENSE](LICENSE) para más detalles.

