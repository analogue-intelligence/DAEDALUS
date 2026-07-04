"""
Geometric renderers. Same points + colours as the blended field, but rendered as hard-edged shapes instead of a smear:

  voronoi_svg  : each word owns a flat-filled Voronoi cell (stained-glass look)
  lowpoly_svg  : Delaunay triangulation, each triangle = average of its corners
  tiles_svg    : a flat rectangle per word on the reading grid (colour blocks)

All pure vector, flat fills -> crisp. The blend lives in render; nothing upstream changes.

Mathematical background:
Voronoi diagram: partitions the plane into regions. The Voronoi cell of point p_i
is the set of all points closer to p_i than to any other generator point:
    V_i = { x in R^2 : ||x - p_i|| <= ||x - p_j|| for all j != i }

Each cell boundary is a segment of the perpendicular bisector between two points.

Delaunay triangulation: the dual of the Voronoi diagram. Connects points into
triangles such that no point lies inside the circumscribed circle of any triangle.
Maximises the minimum angle across all triangles (avoids sliver triangles).
"""
from __future__ import annotations
import numpy as np


def _rgb(c):
    """
    Convert a colour array [r, g, b] in [0, 1] to a hex string '#rrggbb'.
    """
    r, g, b = (np.clip(c, 0, 1) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# Voronoi via per-point half-plane clipping (no scipy geometry needed) 
#
# Instead of computing the full Voronoi diagram analytically, we compute each
# cell independently by starting with the full canvas rectangle and clipping
# it against every other point's half-plane. This is O(N^2) but avoids the
# complexity of a full Voronoi library and handles all edge cases (unbounded
# cells, collinear points) by construction.

#SWITCHING TO THE VORONOI LIBRARY IS FOR LATER


def _seg_isect(p, q, a, b, c):
    """
    Find where the line segment p->q crosses the half-plane boundary a*x + b*y = c.

    The half-plane is defined by the inequality a*x + b*y <= c.
    We're looking for the intersection point of segment (p, q) with the line a*x + b*y = c.

    Uses the signed-distance parameterisation:
        d_p = a*p_x + b*p_y - c   (signed distance of p from boundary)
        d_q = a*q_x + b*q_y - c   (signed distance of q from boundary)

    The crossing parameter t (where 0 <= t <= 1 along the segment) is:
        t = d_p / (d_p - d_q)

    And the intersection point is:
        (p_x + t*(q_x - p_x),  p_y + t*(q_y - p_y))

    Parameters:
    - p, q : tuple (x, y) - Endpoints of the segment.
    - a, b, c : float - Half-plane coefficients: a*x + b*y <= c.

    Returns:
    - (x, y) : tuple of float - The intersection point.
    """
    px, py = p
    qx, qy = q
    dp = a * px + b * py - c
    dq = a * qx + b * qy - c
    t = dp / (dp - dq)
    return (px + t * (qx - px), py + t * (qy - py))


def _clip(poly, a, b, c):
    """
    Clip a convex polygon against the half-plane a*x + b*y <= c.

    Uses the Sutherland-Hodgman algorithm:
        For each edge (prev -> curr) of the polygon:
            - If both vertices are inside: keep curr.
            - If prev is outside, curr is inside: add the intersection, then curr.
            - If prev is inside, curr is outside: add the intersection.
            - If both are outside: skip.

    A small epsilon (1e-9) is added to the boundary test to handle floating-point edge cases where a vertex lies exactly on the boundary.

    Parameters:
    - poly : list of (x, y) - Vertices of the convex polygon in order.
    - a, b, c : float - Half-plane: keep the region where a*x + b*y <= c.

    Returns:
    - list of (x, y) - Vertices of the clipped polygon (may be empty if fully clipped away).
    """
    out = []
    n = len(poly)
    for i in range(n):
        cur, prv = poly[i], poly[i - 1]
        cur_in = a * cur[0] + b * cur[1] <= c + 1e-9
        prv_in = a * prv[0] + b * prv[1] <= c + 1e-9
        if cur_in:
            if not prv_in:
                out.append(_seg_isect(prv, cur, a, b, c))
            out.append(cur)
        elif prv_in:
            out.append(_seg_isect(prv, cur, a, b, c))
    return out


def _cell(i, P, width, height):
    """
    Compute the Voronoi cell for point P[i] by half-plane clipping.

    Start with the full canvas rectangle as the initial polygon. For every other point P[j], clip the polygon to the half-plane containing P[i]
    (the "P[i] side" of the perpendicular bisector between P[i] and P[j]).

    The perpendicular bisector of P[i] and P[j] has equation:
        a*x + b*y = c
    where:
        a = 2*(x_j - x_i)
        b = 2*(y_j - y_i)
        c = x_j^2 + y_j^2 - x_i^2 - y_i^2

    The half-plane a*x + b*y <= c contains all points closer to P[i] than P[j].

    Derivation: point x is closer to P[i] than P[j] iff:
        ||x - p_i||^2 <= ||x - p_j||^2
        x^2 - 2*x*p_i + p_i^2 <= x^2 - 2*x*p_j + p_j^2
        2*(p_j - p_i) . x <= p_j^2 - p_i^2

    Parameters:
    - i : int - Index of the point whose Voronoi cell to compute.
    - P : ndarray (N, 2) - All generator points.
    - width, height : int - Canvas dimensions (defines the bounding rectangle).

    Returns:
    - list of (x, y) - Vertices of the Voronoi cell polygon.
    """
    poly = [(0, 0), (width, 0), (width, height), (0, height)]
    xi, yi = P[i]
    for j in range(len(P)):
        if j == i:
            continue
        xj, yj = P[j]
        # perpendicular bisector coefficients
        a, b = 2 * (xj - xi), 2 * (yj - yi)
        c = (xj * xj + yj * yj) - (xi * xi + yi * yi)
        poly = _clip(poly, a, b, c)
        if not poly:
            break  # cell is empty (shouldn't happen with distinct points)
    return poly


def voronoi_svg(positions, colors, width, height, stroke="#ffffff", stroke_w=1.4):
    """
    Render a Voronoi diagram as an SVG string.

    Each word's Voronoi cell is filled with that word's colour, producing a stained-glass aesthetic. White strokes between cells make the boundaries
    visible.

    Parameters: 
    - positions : ndarray (N, 2) - Word positions (Voronoi generator points).
    - colors : ndarray (N, 3) - Word colours in [0, 1].
    - width, height : int - Canvas dimensions.
    - stroke : str or None - Stroke colour for cell boundaries. None for no stroke.
    - stroke_w : float - Stroke width in pixels.

    Returns:
    - str - SVG markup.
    """
    P = np.asarray(positions, float)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for i in range(len(P)):
        poly = _cell(i, P, width, height)
        if len(poly) < 3:
            continue  # degenerate cell (< 3 vertices = not a polygon)
        # build SVG path: M = move to, L = line to, Z = close
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in poly) + " Z"
        s = (f' stroke="{stroke}" stroke-width="{stroke_w}" stroke-linejoin="round"'
             if stroke else "")
        parts.append(f'<path d="{d}" fill="{_rgb(colors[i])}"{s}/>')
    parts.append("</svg>")
    return "\n".join(parts)


# Low-poly via Delaunay 

def lowpoly_svg(positions, colors, width, height, stroke=None, stroke_w=0.6):
    """
    Render a Delaunay triangulation as a low-poly SVG.

    Each triangle is filled with the average colour of its three corner words, producing a faceted / low-poly look. Canvas corners are added as extra
    points (coloured by their nearest word) so triangles extend to the edges.

    Parameters:
    - positions : ndarray (N, 2) - Word positions (triangle vertices).
    - colors : ndarray (N, 3) - Word colours in [0, 1].
    - width, height : int - Canvas dimensions.
    - stroke : str or None - Explicit stroke colour. If None, each triangle's stroke matches its fill (hiding seams via a hairline same-colour stroke).
    - stroke_w : float - Stroke width in pixels.

    Returns:
    - str - SVG markup.
    """
    from scipy.spatial import Delaunay
    P = np.asarray(positions, float)
    C = np.asarray(colors, float)

    # add the four canvas corners as extra generator points.
    # colour each corner by the nearest word's colour (so edge triangles blend).
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    cc = []
    for cnr in corners:
        # find nearest word to this corner: argmin of squared Euclidean distance
        cc.append(C[np.argmin(((P - cnr) ** 2).sum(1))])
    P2 = np.vstack([P, corners])   # (N+4, 2)
    C2 = np.vstack([C, cc])        # (N+4, 3)

    # compute Delaunay triangulation of the augmented point set
    tri = Delaunay(P2)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for s in tri.simplices:
        # triangle colour = mean of its three vertex colours
        col = C2[s].mean(0)
        pts = " ".join(f"{P2[v,0]:.1f},{P2[v,1]:.1f}" for v in s)
        if stroke:
            st = f' stroke="{stroke}" stroke-width="{stroke_w}"'
        else:
            # hairline stroke in the same colour as the fill hides seam artifacts
            st = f' stroke="{_rgb(col)}" stroke-width="0.6"'
        parts.append(f'<polygon points="{pts}" fill="{_rgb(col)}"{st}/>')
    parts.append("</svg>")
    return "\n".join(parts)


# tile grid (colour blocks in reading order) 

def tiles_svg(rects, colors, width, height, gap=6, radius=2):
    """
    Render a grid of coloured rectangles (one per word in reading order).

    Each rectangle represents a word's colour as a flat block. The gap between tiles gives a mosaic / tile-grid look.

    Parameters:
    - rects : list of (x, y, w, h) - Bounding rectangles for each word (from grid_field.grid_positions or a custom layout).
    - colors : ndarray or list - Colours in [0, 1] for each tile, same order as rects.
    - width, height : int - Canvas dimensions.
    - gap : float - Gap between tiles in pixels. Default 6.
    - radius : float - Corner radius for rounded rectangles. Default 2.

    Returns:
    - str - SVG markup.
    """
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for (x, y, w, h), col in zip(rects, colors):
        # inset each tile by gap/2 on all sides to create the inter-tile gap
        parts.append(f'<rect x="{x+gap/2:.1f}" y="{y+gap/2:.1f}" '
                     f'width="{max(w-gap,1):.1f}" height="{max(h-gap,1):.1f}" '
                     f'rx="{radius}" fill="{_rgb(col)}"/>')
    parts.append("</svg>")
    return "\n".join(parts)