from vit.layers.cls_token import get_cls_token


def test_get_cls_token():
    embed_dim = 768
    cls_token = get_cls_token(embed_dim)
    assert cls_token.shape == (1, 1, embed_dim)
    assert cls_token.requires_grad
