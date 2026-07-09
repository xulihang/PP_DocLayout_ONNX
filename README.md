# PP_DocLayout_ONNX

PaddlePaddle's DocLayout using ONNX. PPDocLayout v3 and PPDocLayout L are supported.

How to use in [ImageTrans](https://www.basiccat.org/zh/imagetrans/):

* PPDocLayout V3

   1. Create a folder named `ppdoclayout` under the `models` folder in ImageTrans's root.
   2. Create a new model.json file under the `ppdoclayout` folder.

       ```json
       {
           "width":800,
           "height":800,
           "use_ppdoclayout":true,
           "resize":false,
           "confidence":0.3,
           "ppdoclayout_version":"v3",
           "support_reading_order":true,
           "model":"PP-DocLayoutV3.onnx"
       }
       ```
       
   3. Download the model PP-DocLayoutV3.onnx and PP-DocLayoutV3.onnx.data from [huggingface](https://hf-mirror.com/Bei0001/PP-DocLayoutV3-ONNX/tree/main) and put the files under the `ppdoclayout` folder.

* PPDocLayout L (This is model is old. Use it for ImageTrans before 6.0.1)

   1. Create a folder named `ppdoclayout` under the `models` folder in ImageTrans's root.
   2. Create a new model.json file under the `ppdoclayout` folder.

       ```json
       {
         "width":800,
         "height":800,
         "use_ppdoclayout":true,
         "model":"pp_doclayout_plus_l.onnx"
       }
       ```
       
   3. Download the model [pp_doclayout_plus_l.onnx](https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/v1.0.0/layout/PP-DocLayout_plus-L/pp_doclayout_plus_l.onnx) and put the file under the `ppdoclayout` folder.

