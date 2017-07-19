Main Idea:<br />
    Randomly take 16 frames to convolute<br />
    Get result by softmaxlayer while the better choice is apply SVM algorithm on outputs of FC6 layer.<br />
Tips<br />
1.Thanks to Kevin Xu/VGG16 and hx173149/C3D-tensorflow<br />
2.hx173149 offer the code to generate the data list<br />
3.hx173149 offer the code the input_data<br />
4.The style of the code learned by Kevin Xu<br />
5.We didn't use ExponentialMovingAverage method to optimize the algorithm<br />
6.It cannot reload sports1m_finetuning_ucf101.model as our pretrained model<br />
7.We use softmax to get the result<br />
