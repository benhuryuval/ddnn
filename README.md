# Noisy inference in Distributed Neural Networks: Project Summary 
Tal Amit and Hila Hoffman, Supervised by Yuval Ben Hur

In data-intensive applications, it is advantageous to perform partial processing close to the data and transmit partial results to the central processor instead of the raw data. When the communication medium is noisy, it is necessary to mitigate the degradation in the model's accuracy.
In this project, we address the issue of reduced accuracy in DDNN models due to noise in the communication channel that transmits information from end devices (to the local aggregator or the cloud). 

This project's objective is to improve the predictor's accuracy under noisy channel conditions by finding an optimal method for selecting aggregation weights. We focus on classification problems for a finite number of classes, where each end device is a Deep Neural Network (DNN) that performs local computation and based on confidence in those computations, it is determined if the sample should exit at that point.

The project aims to find vector α: optimal weights for average pooling aggregation (weighted average), by minimizing the mismatch probability between noisy prediction and the noiseless one. This minimization is accomplished through a gradient descent algorithm. Upon analyzing the results, we successfully developed an algorithm to find these weights and achieved excellent results, establishing this method as a competitive aggregation approach for high performance in local exit.

![image](https://github.com/user-attachments/assets/4104a942-d6a7-4085-81ab-1a1b6ea19033)


# Distributed Deep Neural Networks over the Cloud, the Edge and End Devices
    
Distributed deep neural networks (DDNNs) over distributed computing hierarchies, consisting of the cloud, the edge (fog) and end devices. While being able to accommodate inference of a deep neural network (DNN) in the cloud, a DDNN also allows fast and localized inference using shallow portions of the neural network at the edge and end devices. Due to its distributed nature, DDNNs enhance data privacy and system fault tolerance for DNN applications. When supported by a scalable distributed computing hierarchy, a DDNN can scale up in neural network size and scale out in geographical span. In implementing a DDNN, one is able to map sections of a DNN onto a distributed computing hierarchy. By jointly training these sections, it is possible to minimize communication and resource usage for devices and maximize usefulness of extracted features which are utilized in the cloud. As a proof of concept, it is showen that a DDNN can exploit geographical diversity of sensors to improve recognition accuracy and reduce communication cost. In the experiment, compared with the traditional method of offloading raw sensor data to be processed in the cloud, DDNN locally processes most sensor data on end devices and is able to reduce the communication cost by a factor of over 20x.

## Dependencies

This library is dependent on Python 2.7+ and [Chainer](http://chainer.org/). Please install Chainer 1.17+ before starting.

```
pip install chainer
```

## Further Reading

![CRML_PosterTemplate_Tal_Hila](https://github.com/user-attachments/assets/0b03488c-4471-46ec-a637-0efaa0330015)

