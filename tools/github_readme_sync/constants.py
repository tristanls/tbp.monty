# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import re

IGNORE_DOCS = ["placeholder-example-doc", "some-existing-doc"]
IGNORE_IMAGES = ["docs-only-example.png"]
IGNORE_TABLES = ["example-table-for-docs.csv"]

IGNORE_EXTERNAL_URLS = [
    "openai.com",
    "science.org",
    "annualreviews.org",
    "sciencedirect.com",
]

# Regex for CSV table references
REGEX_CSV_TABLE = re.compile(r"!table\[(.+?)\]")
