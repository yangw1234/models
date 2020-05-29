## Steps

1. follow instructions in https://github.com/yangw1234/models/blob/master/research/object_detection/g3doc/installation.md.
2. follow instructions in https://github.com/yangw1234/models/blob/master/research/object_detection/g3doc/running_pets.md.

   Instead of uploading to google cloud, put all the data and configuration file on a local disk.

   Instead of using faster rcnn, use ssd_mobilenet_v1.
   
   Edit object_detection/samples/configs/ssd_mobilenet_v1_pets.config

3. run the following command

```bash
# from models/research/object_detection
bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] --driver-memory 40g analytics_zoo_main.py
```
