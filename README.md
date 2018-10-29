# Emotion recognition using CNNs
#### The idea is to recognize one of 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from a photo of a person.

We used the fer2013 dataset to train 4 CNNs with different architectures and then stack them using logistic regression.

## Results
By randomly guessing the emotions we expect around 14.3% accuracy (because we have 7 classes). 

Human accuracy is rated at around 65% +- 5%.

State of the art performance is around 71.5% at the time.

Using the stacked model we achieved close to state of the art performance (at the time) - 71%.

## Demo
There is a demo that will try to determine the person's emotion live from the webcam feed. For the purposes of the demo only one of the networks is used, otherwise the video gets choppy due to increased performance requirements.

You can start it by running <code>demo.py</code>

## Training 
We trained all our models on machines in the Google Cloud Platform that have 2 Nvidia K80 GPUs and 100GB ram.

## More info
For graphs detailing the network architectures, examples from the dataset, confusion matrices of the models and other info see our [presentation](https://docs.google.com/presentation/d/1G0BBDb2nK32AsQM5ToPPO_Sx6rpth19IIWsvHD1vkBc/edit?usp=sharing) .

Our work is based on the following papers: [here](https://project.dke.maastrichtuniversity.nl/RAI/wp-content/uploads/2017/05/SMAP_paper_final_version.pdf) , [here](http://cs231n.stanford.edu/reports/2017/pdfs/221.pdf), [here](http://cs231n.stanford.edu/reports/2017/pdfs/224.pdf), [here](https://arxiv.org/pdf/1710.07557.pdf)
