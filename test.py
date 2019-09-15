
import numpy as np
import cv2
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import os

channel = implementations.insecure_channel('localhost', 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'mnist_keras'
request.model_spec.signature_name = 'serving_default'

#### data preprocessing
gray = cv2.imread('/home/zayden/Desktop/samsung/one.png', cv2.IMREAD_GRAYSCALE)
data = gray.astype(np.float32)
data = data[np.newaxis]
data = data[np.newaxis]
train_input = np.reshape(data, (1,28,28,1 ))


request.inputs['input_image'].CopyFrom(make_tensor_proto(train_input, shape=list(train_input.shape)))
result_predict = stub.Predict(request, 1000.0)
print("-----result-----")
print( result_predict)

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
 
final = {}
final['dense_2/Softmax:0'] = list(type_val(result_predict, output_name='dense_2/Softmax:0', output_type='DT_FLOAT' ))
print(final)