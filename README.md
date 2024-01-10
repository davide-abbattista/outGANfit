
# outGANfit - a cDCGANs-based architecture

Conditional Deep Convolutional Generative Adversarial Networks (cDCGANs‐)based architecture to generate outfit items compatible with the input garment, using the Polyvore Outfits dataset. 

## Directory tree

- **architecture**: in this directory the are the python scripts that define the architecture of the Autoencoder and the Generative Adversarial Network;
- **checkpoints**: in this directory there are the saved pre-trained model that can be used to test the model without train it;
- **preprocessing**: in this directory there are all the script used to preprocess out dataset;
- **test_results_images**: in this directory there are some images extracted testing the model;
- **training**: in this directoy there are all the methods written in order to train the autoencoder and the GANs;
- **utility**: in this directory there are the utility methods used during the training phase, such as FID metric, Custom Image Dataset and so on.

## The dataset

Our dataset, Polyvore Outfits (downloadable at [this link](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view)) consists of 261,057 RGB
images of garments measuring 300x300 pixels, divided into the categories “tops”,
“bottoms”, “shoes”, “accessories”, “all-body”, “jewellery”, “bags”, “hats” and
“outerwear”. From these, only the first four were chosen.
The goal is to input a t-shirt to the model and have it generate a matching pant, pair of shoes and accessory (only sunglasses in our case).

## outGANfit architecture

Our architecture includes three different GANs, one for each garment to be predicted (pants, shoes, accessories). For each gan we have a generator and two discriminators, one of which is used to evaluate the compatibility between the conditioning image (t-shirt) and the generated garment. The other discriminator is useful, on the other hand, to classify what is generated as real or fake.
![Architecture](https://i.ibb.co/C5KP874/Screenshot-2024-01-10-alle-15-25-35.png)

## Results

After the evaluation, we saved our best model in a .pth file, named “trained_generator_bottom.pth”. This was made in order to save the updated weights of our model.
We loaded the model and the “testloader”, i.e. iterator of test images, and we tested it.
We obtained a FID value of 175.4166, in accordance with our expectations.
Below, some outputs:
![Results](https://i.ibb.co/nfD6t2s/Screenshot-2024-01-10-alle-15-37-09.png)

## Documentation

You can read the complete documentation at [this link](https://drive.google.com/file/d/1PjbH4C5pEDCSzFhB6YGW-dFH3mL-3fwI/view?usp=sharing)


## Authors

- [Davide Abbattista](https://www.github.com/davide-abbattista)
- [Giovanni Silvestri](https://www.github.com/vannisil)

