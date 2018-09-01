## What does project do? 
This project allow you to generate new names of books and authors . Also create plot:

![Plot](https://thumbs.gfycat.com/TenseSourKitfox-size_restricted.gif)

and some amount of pie charts.

In project you can find some data manipulations and see interesting insights. Such this:

Do you know, that 50% of bestsellers lasted no more than one week in the top?

Just one book could hold on top for more than 100 weeks! And this book is a "The girl on the train" by Paula Hawkins

The largest number of bestsellers were published in the publishing house Grand Central and so on!

Plots and generator based on New York Times bestsellers dataset (from 2011 to 2018 year, not all the time!). Dataset available to review and download [here](https://data.world/) or check `books_uniq_weeks.csv` in repo. 
## How is it set up? 
To run this repo you should have:
```
python3
pandas
numpy
plotly
```
1. Download repo:
```
wget https://github.com/Oysiyl/Bestsellers-New-York-Times.git
```
2. Go to folder:
```
cd Bestsellers-New-York-Times
```
3. Install dependencies:
```
pip3 -r requirements.txt
```
## How is it used? 
After that you can simply run `bestseller.py` from your favorite IDE or terminal:
```
pip3 bestseller.py
```
Thats create a newest version of `dreams.txt` in scripts folder and updated plots. For now you can open this file and have some fun, exploring what generator generate!
Also script create `dreams.txt`, which contains random generated data.
Such this:
```
Your name should be Bella Riley and you are author of GRAY STATE WORLD. This book was published by Ecco/HarperCollins.
```
or this:
```
Your name should be Raine Kearsley and you are author of DIRTY HEAVEN MYTHOLOGY. This book was published by Valkyrie Press.
```
:stuck_out_tongue_winking_eye:
## Compatible 
Tested on Linux OS `Ubuntu 18.04` with `Python 3.6.5`

## License information
This project have not any license now

