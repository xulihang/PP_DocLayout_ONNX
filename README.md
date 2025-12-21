# PP_DocLayout_ONNX

PaddlePaddle's DocLayout using ONNX

Model link: https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/v1.0.0/layout/PP-DocLayout_plus-L/pp_doclayout_plus_l.onnx


How to use in ImageTrans:

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

3. Download the model [pp_doclayout_plus_l.onnx](https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/v1.0.0/layout/PP-DocLayout_plus-L/pp_doclayout_plus_l.onnx) and put it under the `ppdoclayout` folder.
