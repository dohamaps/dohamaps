
python3 setup.py sdist bdist_wheel && python3 -m twine upload dist/*
rm -rf dohamaps.egg-info*
rm -rf dist*
rm -rf build*
