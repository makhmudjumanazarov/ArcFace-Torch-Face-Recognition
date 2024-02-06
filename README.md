## Requirements

To avail the latest features of PyTorch, we have upgraded to version 1.12.0.

- Install [PyTorch](https://pytorch.org/get-started/previous-versions/) (torch>=1.12.0).
- `pip install -r requirement.txt`.
  
## Testing Face Recognition Models via Streamlit 

- Testing  [dataset](https://drive.google.com/file/d/1jzjnyTGISVAR6Lm1koEg1R7QQ0DaD02F/view?usp=sharing)
- [Models](https://drive.google.com/file/d/1twTeIU-Jw_ob1ruSZ3jlbmEJx0SMNJ5y/view?usp=sharing)
- Testing [Video](https://drive.google.com/file/d/1_5c6tJwhhdTbB4b0-eEW4hGG46ByxA5a/view?usp=sharing)

#### If The face recognition is a pytorch model(model.pt) 

```shell
streamlit run stream.py --server.port 8000
```

#### If The face recognition is a onnx model(model.onnx)

```shell
web-demos/src_recognition streamlit run stream.py --server.port 8000
```
