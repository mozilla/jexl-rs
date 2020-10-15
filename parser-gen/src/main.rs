/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

fn main() {
    lalrpop::Configuration::new()
        .generate_in_source_tree()
        .process()
        .unwrap();
}
