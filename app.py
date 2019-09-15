from flask import Flask,request
from flask_restful import Resource, Api

from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import requests
import argparse
import numpy as np
import json
import os
import cv2

app = Flask(__name__)
api = Api(app)
app.config['JSON_AS_ASCII'] = False


#### Post Input ##
parser = argparse.ArgumentParser()
parser.add_argument("-serving_port", "--serving_port", type=int, default=None,
                help='Serving Port ')
args = parser.parse_args()


# port가 고정되있을 것 ( 임의로 박아넣던가 Flag로 추가하여 포트를 인자로 받는다.)
channel = implementations.insecure_channel('localhost',args.serving_port)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request_tensor = predict_pb2.PredictRequest()

## 추후에 할 경우에 inputs 랑 outputs네임의 config파일을 만들어서 관리해야 한다.!! 
import json
with open('./model_config.json', 'r') as w:
    json_data = w.read()
    dictionary_data = json.loads(json_data)



#### output Type Function ####
def type_val(result, output_name, output_type):
    if output_type =='DT_FLOAT':
        out = result.outputs[output_name].float_val
    elif output_type =='DT_INT64':
        out = result.outputs[output_name].int64_val
    elif output_type =='DT_STRING':
        out = result.outputs[output_name].string_val
        for inx, i in enumerate(out):
            if isinstance(out[inx], bytes):
                out[inx] = i.decode("utf-8")
    else:
        out=1
    return out



################이부분은 모델마다 전처리가 다를 가능성이 높으므로 if 문으로 처리하던지 할 것 . 
def image_preprocessing(model_name, image_path):
    if model_name =='mnist_keras':
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        data = gray.astype(np.float32)
        data = data[np.newaxis]
        data = data[np.newaxis]
        data = np.reshape(data, (1,28,28,1 ))
        return data




class process_inference(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        image_path = json_data['image_path']
        model_name = json_data['model_name']
        for inx, i in enumerate(dictionary_data):
            i['model_name'] == model_name
            break
        model_data = dictionary_data[inx]
        signature_name = model_data['signature_name']
        inputs = model_data['inputs']
        outputs = model_data['outputs']
        outputs_type = model_data['outputs_type']
        image_data = image_preprocessing(model_name, image_path)
        # 예측 !! 
        request_tensor.model_spec.name = model_name
        request_tensor.model_spec.signature_name = signature_name
        request_tensor.inputs[inputs].CopyFrom(make_tensor_proto(image_data, shape=list(image_data.shape)))
        result_predict = stub.Predict(request_tensor, 1000.0)
        print("-----result-----")
        print( result_predict)
        final = {}
        final['결과'] = list(type_val(result_predict, output_name=outputs, output_type=outputs_type ))
        return final

api.add_resource(process_inference,'/interlock/inference')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True, threaded=True)