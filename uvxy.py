from typing import Tuple, cast
from pykeen.nn import Interaction, Embedding
from pykeen.models import ERModel
from pykeen.typing import InductiveMode, HeadRepresentation, TailRepresentation, RelationRepresentation
from torch import FloatTensor, rand_like, randn_like, min, max, where, zeros, ones_like, zeros_like, abs, sigmoid, tensor, cat
from torch import LongTensor
from torch.nn.init import uniform_
from torch.nn.functional import normalize
from torch.cuda import is_available as is_cuda_available

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
    a = max(tensor(0), a)

    a = [
        normalize(ai, dim=-1)
        for ai in a.tensor_split(4, dim=-1)
    ]

    return cat(a, dim=-1) 

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
            ("e", "f",) if with_attention_weights else ("e",)

        self.p = p
        self.constraint_mask = constraint_mask
        self.with_attention_weights = with_attention_weights

    def _prepare_octa(self, r):
        if self.with_attention_weights: octa, a = r
        else: octa = r

        return octa.tensor_split(8, dim=-1)
        
    def _prepare_att(self, r):
        if self.with_attention_weights:
            octa, a = r
            return a.tensor_split(4, dim=-1)
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

        ax, ay, au, av = self._prepare_att(r)

        dist = ax * distx + ay * disty + au * distu + av * distv
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
            initializer=uniform_,
            initializer_kwargs=dict(a=-1,b=1),
        )

        r_kwargs = [
            dict(
                embedding_dim=embedding_dim * 8,
                initializer=self._init_octa,
                constrainer=self._constrain_octa
            )
        ]

        if with_attention_weights:
            r_kwargs.append(
                dict(
                    embedding_dim=embedding_dim * 4,
                    initializer=ones_like, # TODO in-place
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

    def score_hrt(
        self,
        hrt_batch: LongTensor,
        *,
        mode: InductiveMode = None
    ) -> FloatTensor:
        h = hrt_batch[:, 0]
        t = hrt_batch[:, 2]

        nb_rels = hrt_batch.size(-1) - 2
        r = hrt_batch[:, 1:-1].squeeze(-1)

        h, r, t = self._get_representations(h=h, r=r, t=t, mode=mode, nb_rels=nb_rels)
        return self.interaction.score_hrt(h=h, r=r, t=t)

    def _get_representations(
        self,
        h: LongTensor,
        r: LongTensor,
        t: LongTensor,
        *,
        mode: InductiveMode,
        nb_rels: int = 1
    ) -> Tuple:
        # TODO consider inductive mode

        hr = self._get_entity_representation(h, ht="head", mode=mode)
        tr = self._get_entity_representation(t, ht="tail", mode=mode)

        rr = self._get_relation_representation(r, nb_rels=nb_rels)

        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hr, rr, tr)),
        )
    
    def _get_entity_representation(self, e: LongTensor, *, ht: str, mode: InductiveMode):
        if ht == "head":
            indices = self.interaction.head_indices
        elif ht == "tail":
            indices = self.interaction.tail_indices
        else:
            raise ValueError(f"expected 'head' or 'tail', got {ht}")

        representations = self._get_entity_representations_from_inductive_mode(mode=mode)
        representations = [representations[i] for i in indices()]

        return [
            representation(indices=e)
            for representation in representations
        ]

    def _get_relation_representation(self, r: LongTensor, *, nb_rels=1):
        if nb_rels == 1:
            return [
                representation(r)
                for representation in self.relation_representations
            ]

        if self.with_attention_weights:
            raise ValueError("Composition with attention weights not supported")

        representation = self.relation_representations[0]
        octa1 = representation(r[:,0])

        # compose relations
        for i in range(1, nb_rels):
            octa2 = representation(r[:,i])
            octa1 = self._compose(octa1, octa2)

        return octa1

    def _compose(self, octa1, octa2):
        cx1, cy1, cu1, cv1, dx1, dy1, du1, dv1 = self._split_octa(octa1)
        cx2, cy2, cu2, cv2, dx2, dy2, du2, dv2 = self._split_octa(octa2)

        xmin1 = cx1 - dx1
        xmax1 = cx1 + dx1
        ymin1 = cy1 - dy1
        ymax1 = cy1 + dy1
        umin1 = cu1 - du1
        umax1 = cu1 + du1
        vmin1 = cv1 - dv1
        vmax1 = cv1 + dv1

        xmin2 = cx2 - dx2
        xmax2 = cx2 + dx2
        ymin2 = cy2 - dy2
        ymax2 = cy2 + dy2
        umin2 = cu2 - du2
        umax2 = cu2 + du2
        vmin2 = cv2 - dv2
        vmax2 = cv2 + dv2

        xmin3 = max(xmin1, max(xmin2 - umax1, vmin1 - xmax2))
        xmax3 = min(xmax1, min(xmax2 - umin1, vmax1 - xmin2))
        ymin3 = max(ymin2, max(umin2 + ymin1, vmin2 - ymax1))
        ymax3 = min(ymax2, min(umax2 + ymax1, vmax2 - ymin1))
        umin3 = max(ymin2 - xmax1, max(umin2 + umin1, vmin2 - vmax1))
        umax3 = min(ymax2 - xmin1, min(umax2 + umax1, vmax2 - vmin1))
        vmin3 = max(xmin1 + ymin2, max(umin2 + vmin1, vmin2 - umax1))
        vmax3 = min(xmax1 + ymax2, min(umax2 + vmax1, vmax2 - umin1))

        return self._merge_octa((
            # centers
            (xmax3 + xmin3) / 2,
            (ymax3 + ymin3) / 2,
            (umax3 + umin3) / 2,
            (vmax3 + vmin3) / 2,
            # deltas
            (xmax3 - xmin3) / 2,
            (ymax3 - ymin3) / 2,
            (umax3 - umin3) / 2,
            (vmax3 - vmin3) / 2
            # TODO deal with lambdas?
        ))

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

        # width with std deviation = 0.5
        # FIXME rather expects an exp distribution starting from min_margin
        octa[4] = max(tensor(0.1), randn_like(octa[4]) / 2)
        octa[5] = max(tensor(0.1), randn_like(octa[5]) / 2)
        octa[6] = max(tensor(0.1), randn_like(octa[6]) / 2)
        octa[7] = max(tensor(0.1), randn_like(octa[7]) / 2)

        # for i in range(0,2): octa[i] = zeros_like(octa[i])
        # for i in range(2,4): octa[i] = init_band_center(octa[i])
        # for i in range(4,6): octa[i] = ones_like(octa[i])
        # for i in range(6,8): octa[i] = init_band_width(octa[i])

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

        if is_cuda_available():
            # TODO get device from model?
            wx = wx.cuda()
            wy = wy.cuda()
            wu = wu.cuda()
            wv = wv.cuda()

        return wx, wy, wu, wv