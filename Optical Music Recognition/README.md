# a1

## Part 0

This program can be run by going into the command line in the "part0" directory and typing "python3 remove_noise.py input.png"

In part0, we have a noisy image of Pichu. After converting this image to the frequency domain, we can see that we have some kind of noise (is this the word "Hi"??). We can see that "Hi", in addition to some strong signal at some higher frequencies (the bright tilted areas). All of these high frequency aspects are the reason why the images looks so noisy.

Therefore, we needed it to implement a low pass filter to remove the noises added. We designed a Gaussian filter using the np.hamming function. Then, a multiplication was done in the frequency domain. The radius of the hamming function indicates the threshold of the low pass filter. 

We can see the reults in "lpf.png". Overall, the noise is almost entirely removed. However, since we used a low-pass filter to remove it, the other high frequency aspects of the image were also removed. This causes a slight blur to the image.

## Part 1

### A)	Description:

This program can be run by going into the command line in the "part1" directory and typing "python3 staff-finder.py input.png"

The main objective of this part of the assignment was to find a group of 5 lines (called a staff where each line is called a stave) in the image using hough transform. If the image contains multiple staffs, we return one. In order to accomplish this, we had to follow several steps which have been documented below.

#### B) STEP 1 - Image Processing and Thresholding:

The very first step in line detection using hough transform is to first detect all the edges. The Pillow library does offer a filter to find the edges but we wont be using this filter as it's better suited for 2D objects in the image and lines only have 1 dimension. Instead we do the following.

First we blur the image using the Minfilter(3) filter offered by the pillow library which is nothing but a convolution that makes dark pixels even more prominent. This is necessary as the lines we are trying to detect are dark and therefore using this filter makes those lines more prominent as well. The next step is to smooth out the image obtained after blurring which is done using the SMOOTH filter. This image is then converted to monochrome which leads to each pixel having values between 0 and 255 to indicate if they are dark or light respectively. Lastly we threshold the image to get rid of any noise i.e. pixels with values less than 150 are set to 255 and the rest are set to 0. This last step only leaves the darker objects (including our lines) in the image and the objects are now represented by white pixels which will be used in our hough transform algorithm for voting.

#### C) STEP 2 - Hough Transform and Hough Space:
   
In order to understand how hough transform actually works we decided to visualize the hough space. To do this, we checked for each pixel (x,y) if it was on a given line r, theta and then voted for that line in the accumulator. For each pixel we had to check it's existence on every line. Our lines are a combination of r, theta where r is in the range of [-diagonal of the image : diagonal of the image] and theta is in the range of [-90 degrees : 90 degrees]. For each pixel x, y we plugged in the values x and y and theta (range from -90 to 90 degrees) into the following formula to find the corresponding r.

r = [X * Cos(theta)] + [Y * Sin(theta)]

We then get the value of r. We increment the value of accumulator at r, theta by 1 (Voting process).

Once each pixel has voted for each r, theta in the accumulator we then adjust the votes to their corresponding pixel intensity  to display the hough space (Pixel Value for Hough Space = number of votes for (r, theta) * 255/maximum number of votes).

We were now able to see the hough space.

#### C) STEP 3 - Identifying the Lines:

We first found the best line in the accumulator (highest number of votes). Since the lines are parallel the theta of the best line will be the same as the theta of the other lines. So in that particular theta, we then find the closest second best line (second highest votes). We find the difference between the values of r for the best and the second best line. We now have the distance between the lines. Using this distance we can now look for lines around the best line at intervals of this distance until we find the 4 closest lines to the best line. This gives us a total of 5 lines which forms the staff.

For part 2, this code was modified to find all the lines with at least 70% the number of votes that the best line had in the accumulator, indicating that these lines are prominent lines in the image.

#### C) Challenges:

The biggest challenge was to find the best way to process the images initially. First we were only thresholding the images which was getting rid of noise and giving us an outline but it was also eliminating a lot of the pixels that were part of the lines in images where there was too much noise. We therefore had to find the best filters to apply to the images (MinFilter and Smoothing).

The second challenge was deciding what range to look for neighbouring lines in once we found the best line and also to avoid looking at the same lines repeatedly. A lot of testing allowed to us to figure out these parameters for an implementation that worked for each image.

The third challenge was understanding at the core how hough transform can actually be used for line detection. Visualizing the hough space before working on the solution was a great way to do so.

## Part 2

This program can be run by going into the command line and typing "python3 omr.py input.png".

For this part, we started off by deciding as a group which approach we should take. We played with the idea of using either segmentation to isolate the notes, or using a hough transform to try to find the elliptic shapes. However, we decided to go with a template matching approach in the end.

We started off by implementing the template matching by using normalized cross correlation between the template and the receptive fields. If the cross correlation for any particular spot was higher than our threshold value, that location would be added to our list of locations. This would be done for each of the three templates. Once we had all of these locations, we would draw the results on the image and print them into the txt file. 

After we got the initial algorithm working, we set off on trying to determine the note labels. Since we assume that the staffs are equidistant and horizontal, we just simply found where each of the box locations for the notes are relative to the staffs, and labeled them as their most likely note. This algorithm can classify notes eight additional locations above or below the end of the detected staffs. This was done to ensure that we able able to classify the really high and low notes of the music as well.

We also has to modify the templates a little bit. For both rest templates, they were a little zoomed out so we decided to zoom in a bit to use just the rest without the additional white space. Similarly, the filled-in note was zoomed in too much. Thus, we created a new template from music1.png notes in order to detect them. This significantly increased the accuracy of the model. We also were able to successfully dynamically scale the image using Pillow in order to fit the templates with the space between staffs in the image.

The one issue that we ran into was on how to choose a proper threshold value. Our algorithm uses an input "k" as a multiplier to either raise or lower the threshold of what is or is not considered a template match. This became a slight issue since we needed a slightly lower threshold on noisy images compared to non-noisy images. We initially tried to just smooth the whole image, but that just decreased the accuracy for all cases. After a bit of messing around with the smoothing parameters there, we abandoned that idea. 

We then also played with the idea of just running the algorithm several times with increasing k values until a good threshold was reached. However, this presented two problems. First, it would make the program run very very slow. Second, there is no real efficient way to choose a good stopping criteria for the looping as this task is non-supervised. Thus, we dropped that idea too.

What we ended up deciding to do is to have a separate, general k value for each template. We found that k=2 for the quarter rest, k=1.75 for the half rest, and k=4 for the filled-in note produced the best results for the most templates, so this is what we are turning in. The one exception to this is in music4.png. Here, we used a lower threshold of k=3 for the notes as the image quality is very very poor in this image. 

Overall, we believe that our code is working well. It gives perfect results on music1. It only misses a few notes and quarter rests in music2. For music3, it can identify all of the notes perfectly, but is slightly inaccurate on some of the labels since the staffs are not perfectly horizontal. Finally, for music4, it can still detect many notes when a lower threshold is used. We also tested the code on rach.png. It works fairly well here but it does take several minutes to run due to how huge the image is. What we found is that it works very well when the notes are clear and when the staffs are perfectly horizontal. It fails if the staffs are slanted or the notes are too distorted or irregular in some way. 

If we were to continue this project, we would most likely continue exploring ways to dynamically choose a threshold value. If we didn't have to manually choose this for the images, the accuracy of the model would surely improve. We would also try implementing a way to classify notes even if the staffs are not horizontal. 

## Contributions of the Authors

Hasan completed all of part0.

Ryan completed all of part1 and the modified hough transform code for part2.

Santhosh completed the template matching and the note box drawing for part2.

Jacob completed the note labeling and the template scaling for part2.

Everyone also contributed roughly equally to the brainstorming and problem solving steps for the problems.


