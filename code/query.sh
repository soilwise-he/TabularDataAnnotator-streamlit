#!/bin/bash


sparql --query query.sparql --data=all.ttl --results=CSV > all.csv
