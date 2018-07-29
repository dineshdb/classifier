from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
from DNN_multilabel import eval1
from CNN_binary import eval2

app = Flask(__name__)
api = Api(app)

class multilabel(Resource):
    def get(self,name):
        return eval1.categories[eval1.np.argmax(eval1.model.predict([eval1.get_tf_record(name)]))]
		
class binary(Resource):
    def get(self,name):
        x_raw = [name] #input sentence
        x_test = eval2.np.array(list(eval2.vocab_processor.transform(x_raw)))

        # Generate batches for one epoch
        batches = eval2.processData.batch_iter(list(x_test), 1, 1, shuffle=False)


        for x_test_batch in batches:
            batch_predictions = eval2.sess.run(eval2.predictions, {eval2.input_x: x_test_batch, eval2.dropout_keep_prob: 1.0})
            if(batch_predictions==[0]):
                 result = "random intent"
            else:
                 result = "functional intent"
        
        return result
    
api.add_resource(binary, '/binary/<name>') # Route_1
api.add_resource(multilabel, '/multilabel/<name>') # Route_2


if __name__ == '__main__':
     app.run(port='5000')
     