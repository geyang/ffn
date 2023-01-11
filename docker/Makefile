default: build release
build:
	docker build -t model-free .
clean-build:
	docker build -t model-free . --no-cache=true
run:
	docker run model-free
release:
	docker tag model-free episodeyang/model-free
	docker push episodeyang/model-free
