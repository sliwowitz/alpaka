#!/bin/bash

#
# Copyright 2020 Benjamin Worpitz
#
# This file is part of Alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

source ./script/travis/travis_retry.sh

source ./script/travis/set.sh

travis_retry sudo apt-get -y --quiet install --no-install-recommends doxygen graphviz