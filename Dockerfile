FROM balenalib/amd64-ubuntu-python:3.9.13

# create workdir
RUN mkdir app/

# change workdir
WORKDIR /app/

# copy requirements
COPY ./requirements.deploy.txt .

# install dependencies
RUN pip install -r requirements.deploy.txt --no-cache-dir

# copy all
COPY . .

# make the script executable
RUN chmod +x runmyplayer.sh

# expose ports
EXPOSE 5800
EXPOSE 5801

CMD ["python", "main.py"]
