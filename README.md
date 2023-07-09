# linkpred

## GIT Submodule
To run the nbfnet, you need to load a GIT submodule via the following commands: 
```
git submodule init; git submodule update 
```
To update the submodule use
```
git submodule update --remote --rebase
```
## NBFNet container

Use the following commands to build and run NBFNet in a docker container.

```bash
docker build -f docker/nbfnet/Dockerfile . -t nbfnet:latest
docker run -ti --rm --gpus all nbfnet:latest bash
```
