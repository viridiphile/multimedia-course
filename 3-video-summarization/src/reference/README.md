# DSNet reference components

The attention classes, anchor-free network, offset conversion, temporal
IoU and NMS come from the authors' MIT-licensed DSNet implementation,
revision `1804176e2e8b57846beb063667448982273fca89`. This subset uses
relative imports and limits the base model constructor to attention,
the configuration used by the basic checkpoint. `tvsum.yml` is the
original split file. [Upstream source](https://github.com/li-plus/DSNet/tree/1804176e2e8b57846beb063667448982273fca89)
