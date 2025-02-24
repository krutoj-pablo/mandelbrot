.PHONY: all clean cuda openacc

all: cuda openacc

cuda:
	$(MAKE) -C cuda

openacc:
	$(MAKE) -C openacc

clean:
	$(MAKE) -C cuda clean
	$(MAKE) -C openacc clean