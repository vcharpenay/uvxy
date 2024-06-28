from pykeen.nn import Interaction, Embedding
from pykeen.models import ERModel
from torch import FloatTensor, rand_like, randn_like, min, max, where, zeros, ones_like, zeros_like, abs, sigmoid, tensor, cat, stack
from torch.nn.init import xavier_uniform_
from torch.nn.functional import normalize
from torch.cuda import is_available as is_cuda_available

def init_rand_width(d, min_width):
    return min_width + randn_like(d).abs()

def init_band_center(u):
    """
    Initialize the center of a diagonal band within the unit square [-1,1].
    """
    return rand_like(u) * 4 - 2

def init_band_width(du, *, init_width = 0.2):
    """
    Initialize the width of a diagonal band.
    """
    return ones_like(du) * init_width

def constrain_width(d):
    """
    Constrain the width of a band to be positive.
    """
    return max(tensor(0), d)

def constrain_att(a):
    """
    Normalize attention weights (restricted to be positive), per constraint.
    """
    return normalize(abs(a), dim=-1) 

def tighten_octa(octa):
    """
    Tighten octagon (uvxy constraints).
    """
    rx, ry, ru, rv, dx, dy, du, dv = octa

    xmin = rx - dx
    xmax = rx + dx
    ymin = ry - dy
    ymax = ry + dy
    umin = ru - du
    umax = ru + du
    vmin = rv - dv
    vmax = rv + dv

    xmin = max(xmin, max(vmin - ymax, max(ymin - umax, (vmin - umax) / 2)))
    xmax = min(xmax, min(vmax - ymin, min(ymax - umin, (vmax - umin) / 2)))
    ymin = max(ymin, max(umin + xmin, max(vmin - xmax, (umin + vmin) / 2)))
    ymax = min(ymax, min(umax + xmax, min(vmax - xmin, (umax + vmax) / 2)))
    umin = max(umin, max(ymin - xmax, max(vmin - 2 * xmax, 2 * ymin - vmax)))
    umax = min(umax, min(ymax - xmin, min(vmax - 2 * xmin, 2 * ymax - vmin)))
    vmin = max(vmin, max(xmin + ymin, max(umin + 2 * xmin, 2 * ymin - umax)))
    vmax = min(vmax, min(xmax + ymax, min(umax + 2 * xmax, 2 * ymax - umin)))

    return (xmax + xmin) / 2, \
        (ymax + ymin) / 2, \
        (umax + umin) / 2, \
        (vmax + vmin) / 2, \
        (xmax - xmin) / 2, \
        (ymax - ymin) / 2, \
        (umax - umin) / 2, \
        (vmax - vmin) / 2

def dist_linear(p, c, d):
    """
    Calculate the distance of a point (p) to a band (c, w)
    as defined in BoxE/ExpressivE.
    """
    tau = abs(p - c)
    w = 2 * d + 1
    k = d * (w - 1 / w)

    return where(tau < d, tau / w, tau * w - k)

def dist_logistic(p, c, d):
    """
    Calculate the distance of a point (p) to a band (c, w)
    as logistic regression on the boolean function in-band/out-of-band.
    """
    return sigmoid(abs(p - c) - d)

class UVXYInteraction(Interaction):
    """
    Scoring method based on the distance of point (x,y) to octagon (uvxy constraints).
    """

    def __init__(
            self,
            *,
            p=1,
            constraint_mask=(1., 1., 1., 1.),
            with_attention_weights=False,
    ):
        """
        Constructor with parameters for uvxy scoring.

        Arguments:
        :param p: passed to torch.Tensor.norm()
        :param constraint_mask: boolean mask givin whether u, v, x or y constraint apply on which coordinate
        :param with_attention_weights: whether scoring should include trainable attention weights
        """
        super().__init__()

        self.entity_shape = ("d",)
        self.relation_shape = \
            ("e", ("f", "g"),) if with_attention_weights else ("e",)

        self.p = p
        self.with_attention_weights = with_attention_weights

        # TODO get device from model?
        self.constraint_mask = \
            [ c.cuda() for c in constraint_mask ] if is_cuda_available() \
            else constraint_mask

    def _prepare_octa(self, r):
        if self.with_attention_weights: octa, _ = r
        else: octa = r

        return octa.tensor_split(8, dim=-1)
        
    def _prepare_att(self, r):
        if self.with_attention_weights:
            _, a = r            
            return (a[..., i, :] for i in range(4))
        else:
            return 1., 1., 1., 1.

    def _dist(self, p, c, d) -> FloatTensor:
        # return dist_linear(p, c, d)
        return dist_logistic(p, c, d)
    
    def _dist_uvxy(self, x, r, y) -> FloatTensor:
        cx, cy, cu, cv, dx, dy, du, dv = self._prepare_octa(r)

        u = y - x
        v = y + x
        
        wx, wy, wu, wv = self.constraint_mask

        distx = wx * self._dist(x, cx, dx)
        disty = wy * self._dist(y, cy, dy)
        distu = wu * self._dist(u, cu, du)
        distv = wv * self._dist(v, cv, dv)

        if distx.size(1) == 1: distx = distx.tile((distu.size(1), 1))
        if disty.size(1) == 1: disty = disty.tile((distu.size(1), 1))

        ax, ay, au, av = self._prepare_att(r)

        dist = cat((
            ax * distx,
            ay * disty,
            au * distu,
            av * distv
        ), dim=-1)

        return -dist.norm(p=self.p, dim=-1)

    def forward(self, h, r, t) -> FloatTensor:
        return self._dist_uvxy(h, r, t)

class UVXY(ERModel):
    hpo_default = dict(
        embedding_dim=dict(type=int, low=16, high=256, q=16)
        # TODO other HPO defaults
    )

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        p: int = 1,
        constraints: str = "uvxy",
        with_attention_weights: bool = False,
        with_tightening: bool = False,
        **kwargs
    ) -> None:
        """
        Constructor parameterizing an octagon model (with uvxy constraints).

        Arguments:
        :param embedding_dim: nb. of coordinates per entity (and per octagon)
        :param p: passed to torch.Tensor.norm()
        :param constraints: string or sequence of strings specifying what constraints to apply on which coordinate; e.g. "uvxy" (all constraints on all coordinates) or ["u", "v"] (u-constraint on the first half of coordinates and v-constraint on the other half)
        :param with_attention_weights: whether scoring should include trainable attention weights
        :param with_tightening: whether octagons are normalized after each batch, ensuring tight uvxy constraints
        """
        self.with_attention_weights = with_attention_weights
        self.with_tightening = with_tightening

        e_kwargs = dict(
            embedding_dim=embedding_dim,
            initializer=xavier_uniform_
        )

        r_kwargs = [
            dict(
                embedding_dim=embedding_dim * 8,
                # TODO better results without proper init?
                # initializer=self._init_octa,
                # constrainer=self._constrain_octa
            )
        ]

        if with_attention_weights:
            r_kwargs.append(
                dict(
                    shape=(4, embedding_dim),
                    initializer=xavier_uniform_,
                    constrainer=constrain_att
                )
            )

        # TODO check constraints are iterable
        if isinstance(constraints, str): constraints = (constraints,)
        mask = self._build_constraint_mask(constraints, embedding_dim)

        super().__init__(
            interaction=UVXYInteraction,
            interaction_kwargs=dict(
                p=p,
                constraint_mask=mask,
                with_attention_weights=with_attention_weights,
            ),
            entity_representations=Embedding,
            entity_representations_kwargs=e_kwargs,
            relation_representations=Embedding,
            relation_representations_kwargs=r_kwargs,
            **kwargs
        )

    def _split_octa(self, octa):
        octa = octa.tensor_split(8, dim=-1)
        return list(octa)
        
    def _merge_octa(self, octa):
        return cat(octa, dim=-1)

    def _init_octa(self, octa):
        octa = self._split_octa(octa)

        # center in unit square
        octa[0] = rand_like(octa[0]) * 2 - 1
        octa[1] = rand_like(octa[1]) * 2 - 1
        octa[2] = octa[1] - octa[0]
        octa[3] = octa[1] + octa[0]

        min_width = 0.1

        octa[4] = init_rand_width(octa[4], min_width)
        octa[5] = init_rand_width(octa[5], min_width)
        octa[6] = init_rand_width(octa[6], min_width)
        octa[7] = init_rand_width(octa[7], min_width)

        octa = tighten_octa(octa)

        return self._merge_octa(octa)

    def _constrain_octa(self, octa):
        octa = self._split_octa(octa)

        for i in range(4, 8): octa[i] = constrain_width(octa[i])

        # FIXME if less than 3 constraints, inconsistent
        if self.with_tightening:
            octa = tighten_octa(octa)

        return self._merge_octa(octa)

    def _build_constraint_mask(self, constraints, dim):
        nb_chunks = len(constraints)
        chunk_size = dim // nb_chunks

        wx = zeros(dim)
        wy = zeros(dim)
        wu = zeros(dim)
        wv = zeros(dim)

        for i, chunk in enumerate(constraints):
            start = chunk_size * i
            end = chunk_size * (i+1)
            s = slice(start, end)

            if "x" in chunk: wx[..., s] = 1.
            if "y" in chunk: wy[..., s] = 1.
            if "u" in chunk: wu[..., s] = 1.
            if "v" in chunk: wv[..., s] = 1.

        return wx, wy, wu, wv