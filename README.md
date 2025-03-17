Build (Ubuntu 22.04):
### Install dependencies
#### 1. Update packages and install g++:
```bash
sudo apt update
sudo apt install -y g++ build-essential
```
#### 2. Install Google Test
```bash
sudo apt install -y libgtest-dev
cd /usr/src/googletest
sudo mkdir build
cd build
sudo cmake ..
sudo make
sudo cp lib/*.a /usr/lib/
```
### Build the project
```bash
chmod +x build.sh
bash build.sh
```
