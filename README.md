# ⭐ Star Detection In Ancient Coins. ⭐
Alexander Jannaeus' coins are some of the most important and well-known examples of ancient Jewish coinage. <br>
Some of his coins feature a star, and there were two types of stars, differing in the number of rays - six or eight. <br>
We made a python program to distinguish between six or eight rays. <br> 
<br>
[![Vid](https://github.com/Tomi-1997/AJ_Star_Detection/blob/main/thumbnail.png)](https://www.youtube.com/watch?v=e3l-CK-rtFY "Demo") <br>

Seeing how many rays there are seems simple to the naked eye. <br>
![D](https://github.com/Tomi-1997/AJ_Star_Detection/blob/main/example_6.png) <br>
![D](https://github.com/Tomi-1997/AJ_Star_Detection/blob/main/example_8.png) <br>

However, there could be a damaged coin <br>
![D](https://github.com/Tomi-1997/AJ_Star_Detection/blob/main/example_8_eroded.png) <br>

Furthermore, this is the size of the data-set part of six rays. <br>
![D](https://github.com/Tomi-1997/AJ_Star_Detection/blob/main/low_data_size.JPG) <br>
The rest (around 300) are eight rays. <br>
To tackle this problem, we have automated the model generation, producing 20 to 30 models (after heavy filtering) and predicting by majority vote. <br>

In the the video above, we see at first pictures of coins with six rays being fed to the program, and labled correctly. <br>
Then, we see untrained pictures of eight rays, also labled correctly. <br>
