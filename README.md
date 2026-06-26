# IceFinder-ISRO-BAH2026-KalkiCoders

# Introduction
 this is a set of code that works together to find ice on the south pole of the moon, find the best place near it to land the ship and pathfind for the rover to get to the ice.

# Languages and Imports
 it uses two languages python[ for easy of use] and C++[ for fater computation]
 there are few imports that need to be installed for python
 1.rasterio
 2.numpy
 3.matplotlib
 4.scipy
 we need to '''pip install |name of import|''' 

# Data need to run it
 basically there are only three data that are needed from the user side- DEM file, DFSAR[Calibrated data] and avergae illuminnation data.
 cause of git file size limitation the size of files, we have not added the file but will tell you how to put them in
```
  MAIN_FOLDER[holding the whole code]
    └──DFSAR
        └──[all the zipped SAR files downloaded from PRADAN]
    └──PSR
        └──LOLA_DEM tif file
    └──ILLUMI
        └──avg_illumiantion_map of moon
```
 [the file names are hardcoded for the psr and illmination. can be changed as per the need]

# Usage of each file
 `1.DFSAR_icemaper.py`
   it takes the DFSAR data and takes out the lv,lh and incidence. the CPR/DOP can not be done here cause of the polaristaion data not being here.
   it takes both gri as well as sri files. and returns two things one in the bin file[lunar_grid_weight.bin]. one is the if ice is there or not and other is what fraction of that place is ice.<br>
 `2.PSR-crater finder.py`
   it takes the lola data for elevation and finds the psr regions as well as doubly shadowed crater. based on the geometric scaling laws for simple impact craters, we find if something is a crater or just a basin also checks if any secondary lighting hits the crater to make sure doubly shadowed are completely hidden. the data about the psr and doubly shadowed crater is the sent forward as[lunar_psr_dem_grid.bin].<br>
 `3.Final_ice.cpp`
   it is called so as it is the final place for calculation of ice. it takes the psr region and combines it with ice detection done using DFSAR data, gotten from the bin files from last two folders. pixels are filtered in amount of ice in them. not only that it combines these pixels to form clumps[>3px] and it sorts the list in amount of ice in each of them. using this data it creates two bin files [lunar_ice_mask.bin]. it simply contains if a certain pixel is normal, psr_ice or doubly_crater_ice. [lunar_psr_locations.bin] you can say its simply an array of elements as <location<x,y>, amount_of_ice> for both psr region and doubly shadow regions.<br>
 `lander.py`
   this is the final file, it takes data from psr, illimation and uses it to find the best landing places near the location it got from lunar_psr_locations.bin and then create a path for the rover to go from lander to ice. it takes in consideration the elevation as well as sunlight[for the solar panels]. this is doen when it checks the coordinates sees if the path was already found and is placed in [mission_cache.bin], if not do what is written above and put in the data. then, if creates an interface where you can see the path being taken as well as where ice is via [lunar_ice_mask.py]. we can see any of them we want.<br>

`why so many seprate file?` this is done because not all processes are needed at the dame time. DFSAR will be re run only if new files are added. PSR will be done if the resolution is changes[118m->20m]. specially the PSR-crater finder.py takes about 1 hour and 20 minutes to run. the cpp will be only run when the other two are changed. but the lander.py will be runned whenever we want to see the data. we can't wait 2hr everytime to see the change. this makes so that you can run the program even if you just have [lunar.py,lunar_ice_mask.bin,lunar_psr_location.bin]. 
