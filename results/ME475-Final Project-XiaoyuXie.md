# Project description

For a classification problem, we generally use a softmax function to get the predicted score. Although a softmax function can normalize the results, it cannot be used to identify unknown class samples and hard samples. A low predict score for a certain class does not mean an unknown class or a hard sample. In addition, it is hard to understand which part of images have a strong influence on the final prediction, especially for these hard samples. So, I used Gradient-weighted Class Activation Mapping (Grad-CAM) to explain the results.

There are two contributions in this final project. First, I proposed a robust classification approach to identify samples from unknown classes and hard samples. Second, I used Grad-CAM to visualization the attention of neural networks to make the results more explainable. Specifically, I find that apparent misclassifications tend to have a larger attention area and understandable misclassifications tend to have a smaller attention are.

Note that due to the requirement of page numbers, I just include the code for training. Other codes can be found in [github](https://github.com/xiaoyuxie-vico/ExplainableAI).

# Dataset

The cat and dog dataset used in this project is downloaded from [Kaggel](https://www.kaggle.com/tongpython/cat-and-dog). Several images are shown in the below:

The data augmentation in the training set includes random rotation (20), random crop scale (0.8, 1.0), Random Horizontal Flip, Random Affine, Normalization. The test set does not use data augmentation.

The batch size is 32 and the dataset will be shuffled.

# Build and train model


```python
# base model is ResNet50 and change the FC layer
model = torchvision.models.resnet50(pretrained=True).to(device)
model.fc = nn.Sequential(
    nn.Linear(2048,2),
).to(device)
# define loss and optimizer
criterian = nn.CrossEntropyLoss()
optimizers = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```


```python
print('[INFO] training')
n_epochs = 20

loss_history, acc_history = [], []
for epoch in tqdm_notebook(range(n_epochs)):
    model.train()
    correct, total = 0., 0.
    for batch_idx, (data, labels, _) in tqdm_notebook(enumerate(train_dataloader)):
        data = data.to(device)
        labels = labels.to(device)
        optimizers.zero_grad()
        output = model(data)
        loss = criterian(output,labels)
        loss.backward()
        optimizers.step()
        
        pred_scores = F.softmax(output, dim=1)
        pred_labels = torch.max(pred_scores, axis=1)[1]
        correct += (pred_labels==labels.long()).sum().item()
        total += labels.shape[0]
        acc = round(float(correct)/total, 4)
        
        loss_history.append(loss.item())
        acc_history.append(acc)
            
        if batch_idx % 20 == 0:
            print('epoch: {}, batch_idx: {}, loss: {}, acc: {}'.format(
                epoch, batch_idx, round(loss.item(), 4), acc))
    if acc > 0.95:
        torch.save(model.state_dict(),'./models/resnet50-epoch{}-Acc{}.h5'.format(epoch, int(acc*10000)))
```

    [INFO] training



    HBox(children=(FloatProgress(value=0.0, max=20.0), HTML(value='')))



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 0, batch_idx: 0, loss: 0.8079, acc: 0.375
    epoch: 0, batch_idx: 20, loss: 0.2993, acc: 0.6994
    epoch: 0, batch_idx: 40, loss: 0.1262, acc: 0.83
    epoch: 0, batch_idx: 60, loss: 0.0862, acc: 0.8678
    epoch: 0, batch_idx: 80, loss: 0.1332, acc: 0.88
    epoch: 0, batch_idx: 100, loss: 0.1579, acc: 0.8936
    epoch: 0, batch_idx: 120, loss: 0.0551, acc: 0.9062
    epoch: 0, batch_idx: 140, loss: 0.0701, acc: 0.9158
    epoch: 0, batch_idx: 160, loss: 0.0389, acc: 0.9216
    epoch: 0, batch_idx: 180, loss: 0.2492, acc: 0.9252
    epoch: 0, batch_idx: 200, loss: 0.0473, acc: 0.93
    epoch: 0, batch_idx: 220, loss: 0.3588, acc: 0.9328
    epoch: 0, batch_idx: 240, loss: 0.0424, acc: 0.9354
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 1, batch_idx: 0, loss: 0.059, acc: 0.9688
    epoch: 1, batch_idx: 20, loss: 0.124, acc: 0.9598
    epoch: 1, batch_idx: 40, loss: 0.0653, acc: 0.9627
    epoch: 1, batch_idx: 60, loss: 0.1109, acc: 0.9606
    epoch: 1, batch_idx: 80, loss: 0.0398, acc: 0.9633
    epoch: 1, batch_idx: 100, loss: 0.0803, acc: 0.965
    epoch: 1, batch_idx: 120, loss: 0.1241, acc: 0.9636
    epoch: 1, batch_idx: 140, loss: 0.1066, acc: 0.9645
    epoch: 1, batch_idx: 160, loss: 0.0354, acc: 0.9658
    epoch: 1, batch_idx: 180, loss: 0.0665, acc: 0.967
    epoch: 1, batch_idx: 200, loss: 0.038, acc: 0.9674
    epoch: 1, batch_idx: 220, loss: 0.1271, acc: 0.9675
    epoch: 1, batch_idx: 240, loss: 0.0412, acc: 0.9676
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 2, batch_idx: 0, loss: 0.1402, acc: 0.9688
    epoch: 2, batch_idx: 20, loss: 0.0531, acc: 0.9464
    epoch: 2, batch_idx: 40, loss: 0.0554, acc: 0.952
    epoch: 2, batch_idx: 60, loss: 0.049, acc: 0.96
    epoch: 2, batch_idx: 80, loss: 0.0165, acc: 0.9599
    epoch: 2, batch_idx: 100, loss: 0.0787, acc: 0.9592
    epoch: 2, batch_idx: 120, loss: 0.2566, acc: 0.9592
    epoch: 2, batch_idx: 140, loss: 0.0621, acc: 0.9603
    epoch: 2, batch_idx: 160, loss: 0.089, acc: 0.9622
    epoch: 2, batch_idx: 180, loss: 0.0474, acc: 0.9624
    epoch: 2, batch_idx: 200, loss: 0.0453, acc: 0.963
    epoch: 2, batch_idx: 220, loss: 0.1472, acc: 0.9632
    epoch: 2, batch_idx: 240, loss: 0.0244, acc: 0.9629
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 3, batch_idx: 0, loss: 0.0847, acc: 0.9375
    epoch: 3, batch_idx: 20, loss: 0.1979, acc: 0.9673
    epoch: 3, batch_idx: 40, loss: 0.0223, acc: 0.9665
    epoch: 3, batch_idx: 60, loss: 0.0785, acc: 0.9672
    epoch: 3, batch_idx: 80, loss: 0.2462, acc: 0.9699
    epoch: 3, batch_idx: 100, loss: 0.1015, acc: 0.9712
    epoch: 3, batch_idx: 120, loss: 0.0163, acc: 0.9698
    epoch: 3, batch_idx: 140, loss: 0.0351, acc: 0.9699
    epoch: 3, batch_idx: 160, loss: 0.0968, acc: 0.9697
    epoch: 3, batch_idx: 180, loss: 0.1516, acc: 0.9701
    epoch: 3, batch_idx: 200, loss: 0.0669, acc: 0.9703
    epoch: 3, batch_idx: 220, loss: 0.105, acc: 0.9692
    epoch: 3, batch_idx: 240, loss: 0.0096, acc: 0.9695
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 4, batch_idx: 0, loss: 0.0254, acc: 1.0
    epoch: 4, batch_idx: 20, loss: 0.1286, acc: 0.9524
    epoch: 4, batch_idx: 40, loss: 0.0283, acc: 0.9566
    epoch: 4, batch_idx: 60, loss: 0.0803, acc: 0.9621
    epoch: 4, batch_idx: 80, loss: 0.1678, acc: 0.9603
    epoch: 4, batch_idx: 100, loss: 0.0262, acc: 0.9629
    epoch: 4, batch_idx: 120, loss: 0.1569, acc: 0.962
    epoch: 4, batch_idx: 140, loss: 0.1616, acc: 0.9637
    epoch: 4, batch_idx: 160, loss: 0.0036, acc: 0.9651
    epoch: 4, batch_idx: 180, loss: 0.1294, acc: 0.9639
    epoch: 4, batch_idx: 200, loss: 0.078, acc: 0.9628
    epoch: 4, batch_idx: 220, loss: 0.013, acc: 0.9638
    epoch: 4, batch_idx: 240, loss: 0.0285, acc: 0.965
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 5, batch_idx: 0, loss: 0.1092, acc: 0.9688
    epoch: 5, batch_idx: 20, loss: 0.0683, acc: 0.9643
    epoch: 5, batch_idx: 40, loss: 0.0342, acc: 0.9611
    epoch: 5, batch_idx: 60, loss: 0.1717, acc: 0.9636
    epoch: 5, batch_idx: 80, loss: 0.2233, acc: 0.9649
    epoch: 5, batch_idx: 100, loss: 0.0068, acc: 0.9669
    epoch: 5, batch_idx: 120, loss: 0.23, acc: 0.969
    epoch: 5, batch_idx: 140, loss: 0.0431, acc: 0.9696
    epoch: 5, batch_idx: 160, loss: 0.0468, acc: 0.9699
    epoch: 5, batch_idx: 180, loss: 0.0518, acc: 0.9693
    epoch: 5, batch_idx: 200, loss: 0.2472, acc: 0.9663
    epoch: 5, batch_idx: 220, loss: 0.0112, acc: 0.9658
    epoch: 5, batch_idx: 240, loss: 0.0219, acc: 0.9667
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 6, batch_idx: 0, loss: 0.0056, acc: 1.0
    epoch: 6, batch_idx: 20, loss: 0.0128, acc: 0.9792
    epoch: 6, batch_idx: 40, loss: 0.0302, acc: 0.9779
    epoch: 6, batch_idx: 60, loss: 0.2222, acc: 0.9744
    epoch: 6, batch_idx: 80, loss: 0.0407, acc: 0.9691
    epoch: 6, batch_idx: 100, loss: 0.0881, acc: 0.9703
    epoch: 6, batch_idx: 120, loss: 0.0042, acc: 0.9698
    epoch: 6, batch_idx: 140, loss: 0.1397, acc: 0.9703
    epoch: 6, batch_idx: 160, loss: 0.0093, acc: 0.9695
    epoch: 6, batch_idx: 180, loss: 0.035, acc: 0.9706
    epoch: 6, batch_idx: 200, loss: 0.0567, acc: 0.9708
    epoch: 6, batch_idx: 220, loss: 0.0137, acc: 0.9717
    epoch: 6, batch_idx: 240, loss: 0.0499, acc: 0.9712
    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    epoch: 7, batch_idx: 0, loss: 0.2208, acc: 0.875
    epoch: 7, batch_idx: 20, loss: 0.0061, acc: 0.9673
    epoch: 7, batch_idx: 40, loss: 0.0951, acc: 0.9649
    epoch: 7, batch_idx: 60, loss: 0.0329, acc: 0.9652
    epoch: 7, batch_idx: 80, loss: 0.0123, acc: 0.966
    epoch: 7, batch_idx: 100, loss: 0.0222, acc: 0.9657
    epoch: 7, batch_idx: 120, loss: 0.005, acc: 0.9675


Loss and accuracy in the training set are:
<img src='figures/loss.jpg' width="50%">
**Finally, the accuracy in the training set and the test set are 0.9809 and 0.9812 respectively.**

## Error analysis (test set)

Althought the accuracy is high (more than 0.98), the trained model still can make some errors.

True label is cat, but predicted label is dog and the score is high.
<img src='figures/error_cat.jpg' width="35%">

True label is dog, but predicted label is cat and the score is high.
<img src='figures/error_dog.jpg' width="70%">

The Grad-CAM results for these wrong predicted images are:
<img src='figures/cam.jpg' width="70%">

We can find that: (1) Apparent misclassifications tend to have a larger attention area; (2) Understandable misclassifications tend to have a smaller attention area.

# Distribution analysis

To solve the problem about apparent misclassifications and reduce the number of understandable misclassifications, we want to analyze the distribution of logits and scores for different classes.

Below, we analyzed the distribution of logits for cat and dog class. The results show that the absolute value for most of logits are in [2, 7], which means that in our training set it is rarely for the model to make a prediction with a lower logits (around 0) or a higher logit (greater than 7 or less than -7). Thus, it is unresonable to believe the prediction if model give such logits. This is the key observation in this project.

<img src='figures/logits.jpg' width="50%">

Kernel density estimation for logits is:
<img src='figures/logits_density.jpg' width="30%">

## Analyze the robustness of classification using logits kernel density estimation

It is common and unavoidable for a model to predict some unkonwn images, including images from unknown classes. Thus, we downloaded three kind of images including a human face, landscape, and apples, which are shown in the below. All of these classes are not included in the training set. We want our model give a low score for these images. But we can find that for the first and third image the model give a high score, which means the model make some serious misclassifications.

For the fourth and fiveth image, even though they are also come from the Internet, the model can give a good prediciton and high scores. Thus, the trained model have a good generalization.

If we see the average of density, we can find that if we use a thershold of 0.04, these wrong predicitons can be alleviated. This is because our model has not "seen" these classes, it will give a low logits for these images. Then we can use this conclusion to identify the unseen classes images and improve the robustness of the model.

<img src='figures/out_of_dataset.jpg' width="100%">

In the above, we test some images from the Internet. Now, we will analyze the test set. The results are shown below. These results also show that a threshold of 0.04 for the average of density is good enough to elimate the wrong predictions.

<img src='figures/out_of_dataset_1.jpg' width="100%">

# Conclusion

- The proposed image classification approach can identify data out of training set based on logits kernel density estimation;
- The proposed image classification approach can help to improve accuracy by identifying data with low density;
- The proposed approach is explainable and can be understand visually;


```python

```
