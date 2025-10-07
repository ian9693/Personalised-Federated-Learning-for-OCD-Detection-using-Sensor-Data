# ocd-detection

The mental illness Obsessive-Compulsive Disorder (OCD) is characterised by obsessive thoughts and compulsive actions. The latter can occur as repetitive activities to ensure that severe fears do not come true. A diagnosis of the disease is usually very late due to a lack of knowledge and shame of the patient. Nevertheless, early detection can significantly increase the success of therapy.With the development of new wearable sensors, it is possible to recognise human activities. Accordingly, wearables can also be used to identify recurring activities that indicate an OCD. Through this form of an automatic detection system, a diagnosis can be made earlier and thus therapy can be started sooner.Since compulsive behaviour is very individual and varies from patient to patient, this paper deals with personalised federated machine learning models. We first adapt the publicly available OPPORTUNITY dataset to simulate OCD behaviour. Secondly, we evaluate two existing personalised federated learning algorithms against baseline approaches. Finally, we propose a hybrid approach that merges the two evaluated algorithms and reaches a mean area under the precision-recall curve (AUPRC) of 0.954 across clients.

## Training
* **Data Augmentation:**
Augmentation of the OPPORTUNITY Activity Recognition Dataset to fit the binary OCD detection task by inserting repetitions of activities.
```shell
python augment.py /path/to/input /path/to/output --num-repetitions=3 --include-original
```

* **Centralized Training:**
Training of a single model on the centralized data.
```shell
python train_centralized.py /path/to/data /path/to/checkpoints --epochs=50
```

* **Local Training:**
Training of one local model for each client, without knowledge sharing.
```shell
python train_localized.py /path/to/data /path/to/checkpoints --clients-per-round=4 --rounds=50
```

* **Federated Averaging:**
Training of a single shared model using Federated Averaging.
```shell
python train_federated_averaging.py /path/to/data /path/to/checkpoints --clients-per-round=4 --rounds=50
```

* **Federated Learning With Personalization Layers:**
Training of a shared model and personalized client models using Federated Learning with Personalization Layers.
```shell
python train_federated_personal_layers.py /path/to/data /path/to/checkpoints --clients-per-round=4 --rounds=50
```

* **Adaptive Personalized Federated Learning:**
Training of a shared model and personalized client models using Adaptive Personalized Federated Learning.
```shell
python train_federated_model_interpolation.py /path/to/data /path/to/checkpoints --clients-per-round=4 --rounds=50
```

* **Federated Learning With Layer Interpolation:**
Training of a shared model and personalized client models using Federated Learning with Layer Interpolation.
```shell
python train_federated_mixed.py /path/to/data /path/to/checkpoints --clients-per-round=4 --rounds=50
```
