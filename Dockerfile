FROM ubuntu:22.04

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages 
RUN apt update \
    && apt install --no-install-recommends -y python3-pip git zip \
    curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++

# Install essential package 
WORKDIR /usr/src/
COPY requirements.txt ./requirements.txt
# RUN python3 -m pip install -r ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copy code to docker 
COPY . .
EXPOSE 8000
COPY ./model_weight/vgg19_bn-c79401a0.pth /root/.cache/torch/hub/checkpoints/vgg19_bn-c79401a0.pth
# CMD python3 /usr/src/run_model.py
CMD ["uvicorn","app-fastapi:app","--host", "0.0.0.0", "--port", "8000"]

