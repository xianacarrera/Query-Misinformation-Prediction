# Query-Misinformation-Prediction


Este repositorio contiene el código, conjuntos de datos y resultados del proyecto Query Misinformation Prediction, cuyo objetivo es analizar la capacidad de diferentes técnicas para predecir la desinformación presente en resultados de búsqueda de consultas. Este estudio se puede adscribir al ámbito de Query Performance Prediction en un contexto *pre-retrieval*, es decir, donde la única información disponible para la predicción es el texto de las consultas de entrada.

## Instalación

En primer lugar, clonar el repositorio:

```bash
git clone https://github.com/xianacarrera/Query-Misinformation-Prediction
```

A continuación, crear dos nuevos entornos a patir de los archivo `environment_qmp.yml` y `environment_qppmetrics.yml`, cuyos nombres predeterminados son `qmp` y `qppmetrics`, respectivamente:

```bash
conda env create -f environment_qmp.yml
# Para cambiar el nombre del entorno:
# conda env create -f environment_qmp.yml -n nuevo-nombre

conda env create -f environment_qppmetrics.yml
# Para cambiar el nombre del entorno:
# conda env create -f environment_qppmetrics.yml -n nuevo-nombre
```

Por último, activar uno u otro entorno.
```bash
conda activate qmp
conda activate qppmetrics
# Alternativamente:
# conda activate nuevo-nombre
```

## Uso

Una vez completada la instalación, se pueden ejecutar:

* `chatgpt.py`, para generar una puntuación de controversia para las consultas en los conjuntos de datos.
* `qpp_metrics.py`, para ejecutar los predictores clásicos de QPP: avg IDF, max IDF, avg SCQ, max SCQ, avg ICTF y SCS sobre las consultas.
* `query_quality_classifier.py`, para generar una predicción utilizando el score de confianza del modelo [Query Quality Classifier](https://huggingface.co/dejanseo/Query-Quality-Classifier) .

Debido a las diferencias entre la versión de Python necesaria para ejecutar Pyserini con esta implementación y para ejecutar el modelo Query Quality Classifier, el programa `qpp_metrics.py` se debe ejecutar con el entorno `qppmetrics`, mientras que el resto de archivos utlizan la configuración de `qmp`.

## Recursos

Proporcionamos los conjuntos de datos utilizados en el proyecto, así como los resultados de varios métodos de predicción. Ten en cuenta que las puntuaciones de controversia no son reproducibles, ya que fueron generadas por un LLM (GPT-4).

También incluimos las puntuaciones de *compatibility* para cada consulta calculadas con dos sistemas de búsqueda (BM25 y MiniLM-L-12-v2), que se usaron como referencia para calcular los coeficientes de correlación de Pearson, Kendall y Spearman con respecto a nuestras predicciones. Se consideraron también el número de documentos *harmful* en los rankings y las puntuaciones NDCG@K asociadas, pero finalmente se descartaron por su falta de variabilidad.

## Contacto

Para cualquier pregunta o inconveniente, contactar a xiana.carrera@rai.usc.es.

---

This repository contains the code, datasets, and results of the Query Misinformation Prediction project, which aims to analyze the performance of different techniques at predicting the amount of misinformation present in search result queries. This study falls within the scope of Query Performance Prediction in a *pre-retrieval* context, meaning that the only information available for prediction is the text of the input queries.

## Installation

First, clone this repository:

```bash
git clone https://github.com/xianacarrera/Query-Misinformation-Prediction
```

Next, create two new environments from the `environment_qmp.yml` and `environment_qppmetrics.yml` files. Their default names are `qmp` and `qppmetrics`, respectively:

```bash
conda env create -f environment_qmp.yml
# To change the environment name:
# conda env create -f environment_qmp.yml -n new-name

conda env create -f environment_qppmetrics.yml
# To change the environment name:
# conda env create -f environment_qppmetrics.yml -n new-name
```

Finally, activate the appropriate environment:
```bash
conda activate qmp
conda activate qppmetrics
# Alternatively:
# conda activate new-name
```

## Use

Once the installation has been successfully completed, run:
* `chatgpt.py`, in order to generate a controversy score for the queries in the datasets.
* `qpp_metrics.py`, to run the classical QPP predictors avg IDF, max IDF, avg SCQ, max SCQ, avg ICTF and SCS on the queries.
* `query_quality_classifier.py`, to produce a prediction using the confidence score of the [Query Quality Classifier](https://huggingface.co/dejanseo/Query-Quality-Classifier) model.

Due to differences between the Python versions required to run Pyserini with this implementation and the Query Quality Classifier model, the `qpp_metrics.py` scrip must be executed using the qppmetrics environment, while the rest of the files use the `qmp` configuration.

## Resources

We provide the datasets used in this project and the results from various prediction methods. Please note that the controversy scores are not reproducible, as they were generated by an LLM (GPT-4).

We also include the compatibility scores for each query and two search systems (BM25 and MiniLM-L-12-v2), which served as the ground truth for calculating Pearson, Kendall, and Spearman correlation coefficients with our predictions. The number of harmful documents in rankings and the NDCG@K scores involving harmful documents were also considered, but they were ultimately discarded due to their lack of variability.

## Contact
For any questions or issues, feel free to reach out at [xiana.carrera@rai.usc.es](mailto:xiana.carrera@rai.usc.es).

