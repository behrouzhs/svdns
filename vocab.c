//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE   1048576
#define SEED    1159241

#define HASHFN  bitwisehash

typedef struct vocabulary {
    char *word;
    long long count;
} VOCAB;

typedef struct hashrec {
    char *word;
    long long count;
    struct hashrec *next;
} HASHREC;

int verbose = 2; // 0, 1, or 2
long long min_count = 1; // min occurrences for inclusion in vocab
long long max_vocab = 0; // max_vocab = 0 for no limit
char input_filename[512];
char output_filename[512] = "vocab.txt";


/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return *s1 - *s2;
}


/* Vocab frequency comparison; break ties alphabetically */
int CompareVocabTie(const void *a, const void *b) {
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return (scmp(((VOCAB *) a)->word,((VOCAB *) b)->word));
    
}

/* Vocab frequency comparison; no tie-breaker */
int CompareVocab(const void *a, const void *b) {
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 );
    else return 0;
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for ( ; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return (unsigned int)((h & 0x7fffffff) % tsize);
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable() {
    int i;
    HASHREC **ht;
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE );
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL;
    return ht;
}

/* Search hash table for given string, insert if not found */
void hashinsert(HASHREC **ht, char *w) {
    HASHREC     *htmp, *hprv;
    unsigned int hval = HASHFN(w, TSIZE, SEED);
    
    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if (htmp == NULL) {
        htmp = (HASHREC *) malloc( sizeof(HASHREC) );
        htmp->word = (char *) malloc( strlen(w) + 1 );
        strcpy(htmp->word, w);
        htmp->count = 1;
        htmp->next = NULL;
        if ( hprv==NULL )
            ht[hval] = htmp;
        else
            hprv->next = htmp;
    }
    else {
        /* new records are not moved to front */
        htmp->count++;
        if (hprv != NULL) {
            /* move to front on access */
            hprv->next = htmp->next;
            htmp->next = ht[hval];
            ht[hval] = htmp;
        }
    }
    return;
}

/* Read word from input stream. Return 1 when encounter '\n' or EOF (but separate from word), 0 otherwise.
   Words can be separated by space(s), tab(s), or newline(s). Carriage return characters are just ignored.
   (Okay for Windows, but not for Mac OS 9-. Ignored even if by themselves or in words.)
   A newline is taken as indicating a new document (contexts won't cross newline).
   Argument word array is assumed to be of size MAX_STRING_LENGTH.
   words will be truncated if too long. They are truncated with some care so that they
   cannot truncate in the middle of a utf-8 character, but
   still little to no harm will be done for other encodings like iso-8859-1.
   (This function appears identically copied in vocab_count.c and cooccur.c.)
 */
int get_word(char *word, FILE *fin) {
    int i = 0, ch;
    for ( ; ; ) {
        ch = fgetc(fin);
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1;
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (ch == '\n') ungetc(ch, fin); // return the newline next time as document ender
            break;
        }
        if (i < MAX_STRING_LENGTH - 1)
          word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH
    }
    word[i] = 0; //null terminate
    // avoid truncation destroying a multibyte UTF-8 char except if only thing on line (so the i > x tests won't overwrite word[0])
    // see https://en.wikipedia.org/wiki/UTF-8#Description
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

int get_counts() {
    long long i = 0, j = 0, vocab_size = 12500;
    // char format[20];
    char str[MAX_STRING_LENGTH + 1];
    HASHREC **vocab_hash = inithashtable();
    HASHREC *htmp;
    VOCAB *vocab;
    FILE *fid = fopen(input_filename, "r");
	if (fid == NULL) {fprintf(stderr,"Unable to open input file %s.\n",input_filename); return 1;}
    
    fprintf(stderr, "BUILDING VOCABULARY\n");
    if (verbose > 1) fprintf(stderr, "Processed %lld tokens.", i);
    // sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    while ( ! feof(fid)) {
        // Insert all tokens into hashtable
        int nl = get_word(str, fid);
        if (nl) continue; // just a newline marker or feof
        if (strcmp(str, "<unk>") == 0) {
            fprintf(stderr, "\nError, <unk> vector found in corpus.\nPlease remove <unk>s from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' > text8.new)");
            return 1;
        }
        hashinsert(vocab_hash, str);
        if (((++i)%100000) == 0) if (verbose > 1) fprintf(stderr,"\033[11G%lld tokens.", i);
    }
	fclose(fid);
    if (verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", i);
    vocab = malloc(sizeof(VOCAB) * vocab_size);
    for (i = 0; i < TSIZE; i++) { // Migrate vocab to array
        htmp = vocab_hash[i];
        while (htmp != NULL) {
            vocab[j].word = htmp->word;
            vocab[j].count = htmp->count;
            j++;
            if (j>=vocab_size) {
                vocab_size += 2500;
                vocab = (VOCAB *)realloc(vocab, sizeof(VOCAB) * vocab_size);
            }
            htmp = htmp->next;
        }
    }
    if (verbose > 1) fprintf(stderr, "Counted %lld unique words.\n", j);
    if (max_vocab > 0 && max_vocab < j)
        // If the vocabulary exceeds limit, first sort full vocab by frequency without alphabetical tie-breaks.
        // This results in pseudo-random ordering for words with same frequency, so that when truncated, the words span whole alphabet
        qsort(vocab, j, sizeof(VOCAB), CompareVocab);
    else max_vocab = j;
    qsort(vocab, max_vocab, sizeof(VOCAB), CompareVocabTie); //After (possibly) truncating, sort (possibly again), breaking ties alphabetically
    
	FILE *fid_out = fopen(output_filename, "w");
	if (fid == NULL) {fprintf(stderr,"Unable to open output file %s for writing.\n", output_filename); return 1;}
    for (i = 0; i < max_vocab; i++) {
        if (vocab[i].count < min_count) { // If a minimum frequency cutoff exists, truncate vocabulary
            if (verbose > 0) fprintf(stderr, "Truncating vocabulary at min count %lld.\n",min_count);
            break;
        }
        fprintf(fid_out, "%s %lld\n", vocab[i].word, vocab[i].count);
    }
	fclose(fid_out);
    
    if (i == max_vocab && max_vocab < j) if (verbose > 0) fprintf(stderr, "Truncating vocabulary at size %lld.\n", max_vocab);
    fprintf(stderr, "Using vocabulary of size %lld.\n\n", i);
    return 0;
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

void print_usage() {
	printf("Simple tool to extract unigram counts\n");
	printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
	printf("Usage options:\n");
	printf("\t-verbose <int>\n");
	printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
	printf("\t-max-vocab <int>\n");
	printf("\t\tUpper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.\n");
	printf("\t-min-count <int>\n");
	printf("\t\tLower limit such that words which occur fewer than <int> times are discarded.\n");
	printf("\nExample usage:\n");
	printf("./vocab_count -verbose 2 -max-vocab 100000 -min-count 10 -input corpus.txt -output vocab.txt\n");
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        print_usage();
        return 0;
    }
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-max-vocab", argc, argv)) > 0) max_vocab = atoll(argv[i + 1]);
    if ((i = find_arg((char *)"-min-count", argc, argv)) > 0) min_count = atoll(argv[i + 1]);
	if ((i = find_arg((char *)"-output", argc, argv)) > 0) strcpy(output_filename, argv[i + 1]);
	
	if ((i = find_arg((char *)"-input", argc, argv)) > 0) 
		strcpy(input_filename, argv[i + 1]);
	else {
		print_usage();
        return 0;
	}
	
    return get_counts();
}

