# Spectral Word Embedding with Negative Sampling

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is submitted to NIPS 2017 and is under review.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites for input preparation

The input to this algorithm is a word-word co-occurrence matrix. For calculating this co-occurrence matrix, we use existing software from GloVe which can be downloaded at:

* [vocab_count](https://github.com/stanfordnlp/GloVe/blob/master/src/vocab_count.c) - This file is used to scan the corpus and build a vocabulary.
* [cooccur](https://github.com/stanfordnlp/GloVe/blob/master/src/cooccur.c) - This file is used, given a vocabulary, to calculate the word-word co-occurrence matrix.

## Compiling the source code

The source code of the software is written in C and can be compiled using standard C compilers in any operating system (Linux, Windows, and macOS). To compile the prerequisites use:

```
$ gcc -Wall -m64 -O3 vocab_count.c -o vocab_count -lm -lpthread
$ gcc -Wall -m64 -O3 cooccur.c -o cooccur -lm -lpthread
```

You can ignore `-Wall` (show all warnings), `-m64` (compile for 64-bit system), `-O3` (optimization level 3). However, `-lm` (link math library) and `-lpthread` (multi-threading library) are required to compile and run the program.

To compile our program run:

```
$ gcc -Wall -fopenmp -m64 -O3 svdns.c -o svdns -lm
```

Our program uses OpenMP shared memory multi-threading library which is standard and is implemented in almost every C compiler. If you ignore `-fopenmp` switch, it will run on a single thread, however, for better performance use this option.

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
