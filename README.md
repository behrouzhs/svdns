# Spectral Word Embedding with Negative Sampling

This software learns a word embedding from the input co-occurrence matrix (preferably extracted from a large corpus such as Wikipedia). This work is submitted to NIPS 2017 and is under review.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites for input preparation

The input to this algorithm is a word-word co-occurrence matrix. For calculating this co-occurrence matrix, we use existing software from GloVe which can be downloaded at:

* [vocab_count](https://github.com/stanfordnlp/GloVe/blob/master/src/vocab_count.c) - This file is used to scan the corpus and build a vocabulary.
* [cooccur](https://github.com/stanfordnlp/GloVe/blob/master/src/cooccur.c) - This file is used, given a vocabulary, to calculate the word-word co-occurrence matrix.

### Compiling the source code

The source code of the software is written in C and can be compiled using standard C compilers in any operating system (Linux, Windows, and macOS). To compile the prerequisites use:

`gcc -Wall -m64 -O3 vocab_count.c -o vocab_count -lm -lpthread`
`gcc -Wall -m64 -O3 vocab_count.c -o vocab_count -lm -lpthread`

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

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
