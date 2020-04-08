# Models for COVID-19 X-ray images classification

## Model 1
### **Dataset constructe**
- 50% COVID-19 X-ray images
- 50% normal healthy images

### **Deep learning model**
- VGG16

### **Result**
- Very accurate to detect COVID-19 

![avatar](/performance1.png)

### Next work
- Change the dataset, where the negative dataset contains other illness

## Model 2 (change the propotation of datasets)
### **Dataset constructe**
- Postive dataset
    - 99 COVID-19 X-ray images
- Negative dataset
    - 47 other lung illness X-ray images
    - 43 normal lung X-ray images
 
### **Result**
- Only 50% accuracy

![avatar](/performance2.png)


## Model 3 (add more COVID-19 images)
### **Dataset constructe**
- Postive dataset
    - **131** COVID-19 X-ray images
- Negative dataset
    - **52** other lung illness X-ray images
    - **80** normal lung X-ray images
 
### **Result**
- ~80% accuracy

![avatar](/performance3.png)
