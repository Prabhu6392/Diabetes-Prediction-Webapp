import numpy as np
import pickle

#loading the model

loaded_model = pickle.load(open('C:/Users/sriva/PycharmProjects/pythonProject/Diabaties Prediction System/model.sav','rb'))

input_data = (1,189,60,23,846,30.1,0.398,59)

# changing data into np array

input_data_nparray = np.asarray(input_data)

#reshape array as we predict for only 1 instance

input_data_reshaped = input_data_nparray.reshape(1, -1)

# standarize the data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('Not Diabatic')
else:
    print('Diabatic')