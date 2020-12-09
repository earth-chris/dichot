# Dichot

`dichot` is set of tools for imaging-spectroscopy-based species classification.

It's name is a twofold botany reference: it's an abbrebiation for dichotomous, referring to the decision tree approach to taxonomic classification; and it's a homonym for dicot, itself an abbreviation for dicotyledons, which is a group of flowering plants whose seeds have two embryonic leaves (cotyledons).

The software this work is based on is described in *Anderson, 2018*, [The CCB-ID approach to tree species mapping with airborne imaging spectroscopy](https://peerj.com/articles/5666/). It was developed as part of the NEON-NIST [ECODSE](http://www.ecodse.org/) data science evaluation competition.

## Introduction

`dichot` can be used in two ways. First, you can run the scripts for training and applying species classification models (under `bin/dc-train` and `bin/dc-apply` respectively). Second, you could import the underlying python functions used in these scripts using `import dichot` (based on the functions in the `dichot/` directory.

```sh
dc-train -i /path/to/training_data -o /path/to/model
dc-apply -i /path/to/testing_data -m /path/to/model -o /path/to/predictions
```

Run `dc-train -h` and `dc-apply -h` to review command line options.

These scripts are intended to work with csv and raster data inputs. HDF support is planned. However, support for raster-based data is currently limited (hdf support is even more so). Please let me know if this is something you would use and I can get my `[redacted]` together.

## ECODSE results

You can reproduce the results submitted to the ECODSE competition by using the `-e` flag in `dc-train` and `dc-apply`. To do this, run the following commands.

```sh
dc-train -o ecodse-model -e -v
dc-apply -m ecodse-model -o ecodse-results.csv -e -v
```

Where the output file `ecodse-results.csv` will have the output species prediction probabilities. The `-e` flag ensure the ECODSE data will be used, and the `-v` flag sets the module to verbose mode to report classification metrics.

Due to some versioning issues, the results are not exactly the same as what was submitted. If you *really* want to find the original results, see the original [scrappy code](https://github.com/christobal54/aei-grad-school/blob/master/bin/neon-classification.py).

## Using other data

The CCB-ID scripts allow using custom data as inputs to model building. These custom data should share the same formats as the data in `support_files/`. Other modifications can be made to the CCB-ID approach, such as using a custom data reducer or custom classification models. This is done by saving these custom objects to a python `pickle` file, then using `ccb-id train` options like `--reducer /path/to/reducer.pck` or `--models /path/to/model1.pck /path/to/model2.pck`. The idea here was to allow you to bring your own data to run new models. Currently, the defaults set to use the NEON/ECODSE data.

## Install


## Contact

All (c) 2018+ Christopher B. Anderson
- [E-mail](mailto:cbanders@stanford.edu)
- [Google Scholar](https://scholar.google.com/citations?user=LoGxS40AAAAJ&hl=en)
- [Personal website](https://earth-chris.github.io/)
