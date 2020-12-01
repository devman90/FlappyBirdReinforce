git clone https://github.com/ntasfi/PyGame-Learning-Environment
mv -f PyGame-Learning-Environment/ple .
rm -rf PyGame-Learning-Environment
mv ple/games/flappybird/assets/background-day.png ple/games/flappybird/assets/background-day-orig.png
mv ple/games/flappybird/assets/background-night.png ple/games/flappybird/assets/background-night-orig.png
cp assets/blank.png ple/games/flappybird/assets/background-day.png
cp assets/blank.png ple/games/flappybird/assets/background-night.png
