#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: David Adarkwah
# DATE CREATED: 20/05/2020                              
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Create list of files in directory
    image_list = listdir(image_dir)
    
    # display the list of images in image_list
    for i in range(0, len(image_list), 1):
        print("{:2d} file: {:>25}".format(i+1, image_list[i]))
    
    # Create an empty dictionary
    dic_results = dict()
    
    # check the number of items in the dictionary
    items_in_dic = len(dic_results)
    print("\nEmpty dictionary results_dic - n items=", items_in_dic)
    
    # Extract pet names from each filename
    for i in range(0, len(image_list), 1):
        if image_list[i][0] != ".":
            names = image_list[i].split("_")
            pet_label = ""
            for name in names:
                if name.isalpha():
                    pet_label += name.lower() + " "
                    
            pet_label = pet_label.strip()
            if image_list[i] not in dic_results:
                dic_results[image_list[i]] = [pet_label]
            else:
                print("** Warning: Key=", image_list[i], "already exists in results_dic with value =", dic_results[image_list[i]])
            
     
    # Replace None with the results_dic dictionary that you created with this
    # function
    return dic_results
