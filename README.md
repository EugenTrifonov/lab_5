# Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
Файл train_all.py

Было использовано несколько техник аугментации данных с оптимальными параметрами 
```python
example['image'] = tf.image.resize(example['image'], tf.constant([250, 250]))
new_image = tf.image.adjust_contrast(image, 2)
new_image = tf.image.adjust_brightness(new_image, 0.4)
new_input=tf.keras.layers.experimental.preprocessing.RandomRotation(0.05,fill_mode='reflect')(inputs)
new_input=tf.keras.layers.GaussianNoise(0.1)(new_input)
```
## Transfer learning
Во всех случаях была экспоненциальная политика изменения темпа обучения с параметрами

![image](https://user-images.githubusercontent.com/80068414/113750074-00e34800-9713-11eb-9a10-afb5f73a58bc.png)
 
 Метрика качества на валидации
 
 ![acc_1](https://github.com/EugenTrifonov/lab_5/blob/main/graph/epoch_categorical_accuracy_transfer.svg)
 
  Функция потерь на валидации
  
  ![loss_1](https://github.com/EugenTrifonov/lab_5/blob/main/graph/epoch_loss_transfer.svg)
 
 ## Fine-tuning
 
 ![image](https://user-images.githubusercontent.com/80068414/113753113-6422a980-9716-11eb-9c23-e9b2cb98cfb7.png)
 
 Метрика качества на валидации
 
  ![acc_2](https://github.com/EugenTrifonov/lab_5/blob/main/graph/epoch_categorical_accuracy_fine_tuning.svg)
  
   Функция потерь на валидации
   
  ![loss_2](https://github.com/EugenTrifonov/lab_5/blob/main/graph/epoch_loss_fine_tuning.svg)
  
  ## Анализ результатов 
