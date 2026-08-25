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
   4. For macOS, you can use the mlx version which is faster: <https://github.com/xulihang/PP_DocLayout_MLX>

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

## Filter Results

If you only need to use specific classes in ImageTrans, you can define them in JSON.

`class_names_to_use` defines classes to use for balloon detection and `class_names_to_use_for_panel` defines classes to use for panel detection.

Sample JSON:

```json
{
  "width":800,
  "height":800,
  "use_ppdoclayout":true,
  "confidence":0.3,
  "resize":false,
  "support_reading_order":true,
  "ppdoclayout_version":"v3",
  "class_names_to_use": [
    "abstract",
    "algorithm",
    "aside_text",
    "content",
    "display_formula",
    "document_title",
    "figure_title",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote"
  ],
  "class_names_to_use_for_panel": [
    "abstract",
    "algorithm",
    "aside_text",
    "content",
    "display_formula",
    "document_title",
    "figure_title",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "text",
    "vertical_text",
    "vision_footnote"
  ],
  "model":"PP-DocLayoutV3.onnx"
}
```
