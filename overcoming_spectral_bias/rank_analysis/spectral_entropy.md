Inspect $\gamma$
```python
for entry in logger.load_pkl('data/spectrum_gamma.pkl'):
    key, = entry.keys()
    spectrum, = entry.values()
    rank = spectral_entropy(spectrum)
    doc.print(key, rank)
```

```
0.99 7.45149109647751
0.9 7.277962463854284
0.6 6.649426371585156
0.1 3.4132912577686985
```
Now inspect the Horizon
```python
for entry in logger.load_pkl('data/spectrum_H.pkl'):
    key, = entry.keys()
    spectrum, = entry.values()
    rank = spectral_entropy(spectrum)
    doc.print(key, rank)
```

```
4 6.914774853510327
3 6.053999057150797
2 5.297699522905671
1 1.5206363643434435
```
