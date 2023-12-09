# Structure
```
├── src
│   ├── data
│   │   ├── dataset - dataset for testing (1968 files)
│   │   ├── model
│   │   │   ├── model.tflite - model in tflite format
│   │   │   ├── model.hpp - model inbuilt to c++ header (was generated automatically)
│   │   │   ├── model_meta.hpp - configuration and vocabulary (was generated automatically)
│   │   ├── report
│   │   │   ├── test_results_analysis.txt - report after testing
│   ├── libtglang - library for language identification
│   ├── libtglang-tester - test program for library
│   ├── ml - code for training and data processing
│   ├── scripts - build, test and test analysis scripts
├── libtglang.so
```

# Algorithm

## Data preprocessing

1. Process source string:

    * resize to 4096 symbols
    * remove all symbols inside '' and ""
    * find minimal number of leading spaces and shift all text to the left
    * rtrim each line
    * remove empty lines

2. Encode text:
    * tokenise with vocabulary (~2.5x times reduce number of symbols)
    * merge all continious unknown tokens to one unknown (for example all letters and numbers were marked as unknown)
    * if we have more than MAX_SIZE tokens in vector, we cut each line up to N tokens, where N is max(MAX_SIZE/len(line), MAX_LINE_SIZE)
    * if we still have more than MAX_SIZE tokens, we remove lines which starts with the same token as previous line (excluding spaces). Idea here is to remove comments or repeated contructions like variables declaration.
    * if it's still more than MAX_SIZE, we take first MAX_SIZE tokens

### Example:
Source
```c
#include <stdio.h>

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    printf("%d", a + b);
}
```
Decoded text after encoding
```
#include <std<UNK>.<UNK>>
int main() {
    int <UNK>, <UNK>;
    scan<UNK>("", &<UNK>, &<UNK>);
    printf("", <UNK> + <UNK>);
}
```


## Inference

I used pruned 2-layers MobileBERT (https://huggingface.co/docs/transformers/model_doc/mobilebert) with such config:

```yaml
embedding_size: 96
hidden_size: 164
intermediate_size: 164
num_attention_heads: 4
num_feedforward_networks: 2
num_hidden_layers: 2
max_tokens: 128
threshold: 0.2
```

# Testing

- For testing I selected 1968 files from source dataset (~20 per language).
- Environment:
    - AMD Ryzen Threadripper PRO 5965WX
    - Docker image 'debian:10'
    - Docker container with limited cpu (8 cores)
- Testing script measures full program time thourgh inbuilt command '_time_', including library initialization and file reading.
- From experiments the code without prediction takes 0.004s -> ~0.009s is a time for prediction.

|MaxTokens=256|||
|----|----|---|
|__Accuracy__|0.979  | [1926/1968]|
|__Program avg time__|0.020s | [min:  0.003s, max:  0.026s]|
|__Func call avg time__|0.016s | (theoretically) |

|MaxTokens=128|||
|----|----|---|
|__Accuracy__|0.975  | [1918/1968]|
|__Program avg time__|0.013s | [min:  0.003s, max:  0.017s]|
|__Func call avg time__|0.009s | (theoretically) |

|MaxTokens=96|||
|----|----|---|
|__Accuracy__|0.971  | [1911/1968]|
|__Avg time__|0.011s | [min:  0.003s, max:  0.014s]|
|__Func call avg time__|0.007s | (theoretically) |

|MaxTokens=64|||
|----|----|---|
|__Accuracy__|0.957  | [1884/1968]|
|__Avg time__|0.010s | [min:  0.003s, max:  0.014s]|
|__Func call avg time__|0.006s | (theoretically) |

# Training

## Datasets

|Source|Files|
|---|---|
|GitHub|1862768|
|StackOverflow|129122|
|Rosetta|98106|
|DLLD|48463|
|Generated|16000|
|ShortSamples|100|

- Rosetta and DLLD as it is
- For GitHub I implemented parser by language, generated list of extensions for each language (utils/lang_enum.py) through ChatGPT and parsed files by extension.
- For StackOverflow I used dump from https://archive.org/details/stackexchange and only used posts as TGLANG_OTHER label. (Seems quite close to posts in TG)
- For rare languages (FIFT, TL and FUNC) I prepared set of ~100 short snippets for each language and used them to generate variative combinations.
- I didn't implement any sortings by popularity of the language in this solution. Therefore I added some short samples with high weight to overfit model predict this syntax as popular language (for C vs D, JSON vs Other, etc.)

## Vocabulary

- Using ChatGPT I generated set of special chars and keywords for each language (utils/lang_constructs.py) and merged them into one vocabulary.
- Additionally I used top 200 ngrams from all languages.

## TrainConfig

|||
|----|----|
|__Model__|MobileBERT (https://huggingface.co/docs/transformers/model_doc/mobilebert)|
|__Loss__|CrossEntropyLoss with label smoothing (0.15) and with Exponential class frequency weighting|
|__Optimizer__|AdaBelief (WEIGHT_DECAY = 1e-1, BETAS = (0.9, 0.95))|
|__Scheduler__|CosineAnnealing (LR = 2e-3, MIN_LR = 2e-5)|
|__BatchSize__|epochs 1-60: 512, 61-200: 1024, 201-350: 1536, >351: 2048|
|__Augs__|Randomly select 5-100 lines from each file|
|__GradientClip__|2.0|
|__Precision__|MixedPrecision|
|__Epochs__|1100 (~1.2min per epoch for 2xNVIDIA 4090)|
|__MaxTokens__|512|

# Solution

- converted model from pytorch to tflite (torchlib doesn't support static linking)
- quantization to float16 (~1.5 times faster for some cpu)
- model inbuilt directly to library (no need to load weights)

# Known issues

- Short snippets of code 1-4 lines are not classified correctly because of difficulty to prepare such dataset automatically. (not clear how to recognize comments and code for each lang fast...)
- Some snippets of different languages with similar syntax are randomly classifying without taking into account popularity of the language. (e.g. a+b could be classified as D language)

# Languages distribution

|Language|Files Number|
|---|---|
|TGLANG_LANGUAGE_OTHER|            199633|
|TGLANG_LANGUAGE_UNREALSCRIPT|      65160|
|TGLANG_LANGUAGE_MARKDOWN|          64280|
|TGLANG_LANGUAGE_JSON|              57394|
|TGLANG_LANGUAGE_CSHARP|            49373|
|TGLANG_LANGUAGE_GO|                45550|
|TGLANG_LANGUAGE_JAVASCRIPT|        45241|
|TGLANG_LANGUAGE_JAVA|              44745|
|TGLANG_LANGUAGE_CPLUSPLUS|         44402|
|TGLANG_LANGUAGE_VERILOG|           43892|
|TGLANG_LANGUAGE_C|                 41433|
|TGLANG_LANGUAGE_PYTHON|            41416|
|TGLANG_LANGUAGE_RUST|              41050|
|TGLANG_LANGUAGE_TYPESCRIPT|        40550|
|TGLANG_LANGUAGE_SOLIDITY|          40190|
|TGLANG_LANGUAGE_YAML|              38919|
|TGLANG_LANGUAGE_XML|               29076|
|TGLANG_LANGUAGE_ASSEMBLY|          28913|
|TGLANG_LANGUAGE_D|                 28250|
|TGLANG_LANGUAGE_DOCKER|            26225|
|TGLANG_LANGUAGE_SCALA|             24657|
|TGLANG_LANGUAGE_RUBY|              24574|
|TGLANG_LANGUAGE_CLOJURE|           23926|
|TGLANG_LANGUAGE_PHP|               23779|
|TGLANG_LANGUAGE_POWERSHELL|        23695|
|TGLANG_LANGUAGE_SMALLTALK|         23266|
|TGLANG_LANGUAGE_FORTRAN|           23199|
|TGLANG_LANGUAGE_FSHARP|            23080|
|TGLANG_LANGUAGE_IDL|               22410|
|TGLANG_LANGUAGE_HASKELL|           22099|
|TGLANG_LANGUAGE_PERL|              21688|
|TGLANG_LANGUAGE_JULIA|             21503|
|TGLANG_LANGUAGE_CRYSTAL|           21264|
|TGLANG_LANGUAGE_ACTIONSCRIPT|      21218|
|TGLANG_LANGUAGE_ADA|               21186|
|TGLANG_LANGUAGE_KOTLIN|            21146|
|TGLANG_LANGUAGE_LUA|               20942|
|TGLANG_LANGUAGE_SCHEME|            20901|
|TGLANG_LANGUAGE_APACHE_GROOVY|     20851|
|TGLANG_LANGUAGE_OCAML|             20831|
|TGLANG_LANGUAGE_ELIXIR|            20815|
|TGLANG_LANGUAGE_COMMON_LISP|       20812|
|TGLANG_LANGUAGE_SWIFT|             20568|
|TGLANG_LANGUAGE_ERLANG|            20527|
|TGLANG_LANGUAGE_DART|              20163|
|TGLANG_LANGUAGE_PASCAL|            19911|
|TGLANG_LANGUAGE_MATLAB|            19524|
|TGLANG_LANGUAGE_TCL|               19504|
|TGLANG_LANGUAGE_ELM|               19404|
|TGLANG_LANGUAGE_ABAP|              19279|
|TGLANG_LANGUAGE_CMAKE|             19175|
|TGLANG_LANGUAGE_DELPHI|            19152|
|TGLANG_LANGUAGE_VISUAL_BASIC|      18940|
|TGLANG_LANGUAGE_R|                 18330|
|TGLANG_LANGUAGE_VALA|              17963|
|TGLANG_LANGUAGE_QML|               17933|
|TGLANG_LANGUAGE_OPENEDGE_ABL|      17744|
|TGLANG_LANGUAGE_PL_SQL|            17549|
|TGLANG_LANGUAGE_FORTH|             17425|
|TGLANG_LANGUAGE_SHELL|             17317|
|TGLANG_LANGUAGE_NIM|               17161|
|TGLANG_LANGUAGE_OBJECTIVE_C|       17013|
|TGLANG_LANGUAGE_CSS|               16868|
|TGLANG_LANGUAGE_SQL|               16560|
|TGLANG_LANGUAGE_BASIC|             16186|
|TGLANG_LANGUAGE_AUTOHOTKEY|        15798|
|TGLANG_LANGUAGE_SAS|               15534|
|TGLANG_LANGUAGE_PROTOBUF|          15264|
|TGLANG_LANGUAGE_GAMS|              14262|
|TGLANG_LANGUAGE_HTML|              14077|
|TGLANG_LANGUAGE_COFFESCRIPT|       11360|
|TGLANG_LANGUAGE_TL|                10051|
|TGLANG_LANGUAGE_MAKEFILE|           9826|
|TGLANG_LANGUAGE_APEX|               9445|
|TGLANG_LANGUAGE_COBOL|              9081|
|TGLANG_LANGUAGE_BATCH|              8508|
|TGLANG_LANGUAGE_RAKU|               8057|
|TGLANG_LANGUAGE_LATEX|              7742|
|TGLANG_LANGUAGE_PROLOG|             7727|
|TGLANG_LANGUAGE_GRADLE|             7522|
|TGLANG_LANGUAGE_LISP|               7154|
|TGLANG_LANGUAGE_ASP|                5641|
|TGLANG_LANGUAGE_GRAPHQL|            5534|
|TGLANG_LANGUAGE_CSV|                5450|
|TGLANG_LANGUAGE_VBSCRIPT|           5103|
|TGLANG_LANGUAGE_AWK|                4599|
|TGLANG_LANGUAGE_APPLESCRIPT|        3838|
|TGLANG_LANGUAGE_INI|                3831|
|TGLANG_LANGUAGE_NGINX|              3437|
|TGLANG_LANGUAGE_HACK|               3403|
|TGLANG_LANGUAGE_FIFT|               3166|
|TGLANG_LANGUAGE_FUNC|               3092|
|TGLANG_LANGUAGE_ICON|               1770|
|TGLANG_LANGUAGE_1S_ENTERPRISE|      1681|
|TGLANG_LANGUAGE_WOLFRAM|            1627|
|TGLANG_LANGUAGE_BISON|              1101|
|TGLANG_LANGUAGE_LOGO|                982|
|TGLANG_LANGUAGE_REGEX|               824|
|TGLANG_LANGUAGE_TEXTILE|             127|
|TGLANG_LANGUAGE_KEYMAN|              115|
