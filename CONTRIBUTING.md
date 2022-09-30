# Contributing to _flarestack_

Since _flarestack_ is a free and open-source software project, *anyone* can contribute it. You simply need to create a fork of the repository, commit your changes, and then make a pull request.

If you are a member of the IceCube collaboration, you can also ask for permission to merge directly to the master branch. 

In either case, we have a few general guidelines that are helpful for keeping things organised:

### 1. Use Github Issues
Whether you found a bug, you want to request an enhancement, or you're actively developing a new feature, 
Github Issues is a great place to keep everyone informed about what you're working on. 
Click on the label button to provide more info about your topic. 
Every time you make a relevant commit, remember to tag the issue (e.g `git commit -m 'progress on #12'`), 
and when you finish and issue you can close it with a commit too! (e.g `git commit -m 'Close #12`')

### 2. Use Pull Requests
After you checked GitHub Issues and found that no one else is working on your idea, be the hero and do so yourself! Create a branch to track your development and submit a PR when it is ready.

#### Major contribution
Create a new branch and implement your changes and when you're done, 
request review by the current maintainers (you can find who they are in the documentation). 
This ensures that the people in charge know what's going on and can keep an overview over the changes. 
You can expect reviewers to give an answer within about a week. Do not hesitate to contact them if there is no reaction
within that time.

#### Minor changes
If you want to only make minor changes, you should also create a branch and submit a pull request. 
But if your changes are minimal and do not touch core functionality (and you have the GitHub access rights), 
you can also merge your pull request directly. 
If in doubt please do take the detour through the review route!

### 3. Keep the CI happy!
GitHub Actions runs all of our unit tests (also for non-master branches), to make sure you didn't break anything with your commit. 
You can see if travis is happy by checking on the GitHub page 
(look for the badge 
[![CI](https://github.com/icecube/flarestack/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/icecube/flarestack/actions/workflows/continous_integration.yml), 
or a tick/cross next to your commit), or on the [GitHub actions website](https://github.com/icecube/flarestack/actions). 
If your commit failed, be sure to check the logs, to see exactly what went wrong. 
If you have access to the IceCube Slack, 
you will see Travis posting directly in the #flarestack channel after each commit.

### 4. Keep the CI busy!
The CI will only run unit tests if we make the unit tests first. 
When you add a new feature, you also need to add some unit tests so that we can ensure this feature continues to work 
in the future. Your tests should be saved in the `tests/` directory, and you can find plenty of examples there to copy. 
Coveralls.io checks how much of the code is covered by tests, and helps you see which lines still need to be covered. 
You can see all of this on the [website](https://coveralls.io/repos/github/icecube/flarestack) or a summary badge 
[![Coverage Status](https://coveralls.io/repos/github/icecube/flarestack/badge.svg?branch=master)](https://coveralls.io/github/icecube/flarestack?branch=master). 
If your commit adds a lot of new code but does not add unit tests, your commit will be tagged on github with a 
red cross to let you know that the code coverage is decreasing. 
If you want to know more about how to design unit tests, you can check out a guide 
[here](https://medium.com/swlh/introduction-to-unit-testing-in-python-using-unittest-framework-6faa06cc3ee1).

### 5. Keep your analysis reproducible! 
If you are a member of the IceCube Collaboration that is ready for an analysis unblinding, 
you should create a release of the flarestack software beforehand and include it in your analysis wiki. 
This ensures that, in the future, anyone can reproduce your analysis exactly by downloading the same version of 
your software. 
You should also create a `readme.md`/`readme.txt` file in your analysis directory, 
listing the software you used, including all catalogues, 
and giving some brief instructions for how another person would reproduce your results. 
An example can be found [here](https://github.com/icecube/flarestack/blob/master/flarestack/analyses/tde/README.txt).
In general, be sure to follow the most up-to-date reproducibility guidelines of the Collaboration.
While it has been done in the past, storage of data such as catalogues in the flarestack repository is not currently recommended.
