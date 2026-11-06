//! GENI insertion (Gendreau, Hertz & Laporte 1992, types I and II) of a
//! vertex v into one tour, for symmetric distances.
//!
//! A tour is a slice `t` of distinct nodes read as a cycle, t[0] → t[1] → …
//! → t[L−1] → t[0]; positions are taken modulo L, so the moves are the same
//! at the start, the middle and the end of the slice, and the tests check
//! all three. Both orientations are tried: `rev` reads the tour as t[0],
//! t[L−1], …, t[1].
//!
//! With i, j, k, l positions (in the orientation used) and v_x = t[x]:
//!
//! * plain: v between v_i and v_{i+1}.
//! * type I (k on the path v_{j+1} … v_{i−1}): delete (v_i, v_{i+1}),
//!   (v_j, v_{j+1}), (v_k, v_{k+1}); add (v_i, v), (v, v_j), (v_{i+1}, v_k),
//!   (v_{j+1}, v_{k+1}). The paths v_{i+1} … v_j and v_{j+1} … v_k are
//!   reversed: v_i, v, v_j … v_{i+1}, v_k … v_{j+1}, v_{k+1} … v_{i−1}.
//! * type II (l on v_{i+2} … v_j, k on v_{j+2} … v_i): delete (v_i, v_{i+1}),
//!   (v_{l−1}, v_l), (v_j, v_{j+1}), (v_{k−1}, v_k); add (v_i, v), (v, v_j),
//!   (v_l, v_{j+1}), (v_{k−1}, v_{l−1}), (v_{i+1}, v_k):
//!   v_i, v, v_j … v_l, v_{j+1} … v_{k−1}, v_{l−1} … v_{i+1}, v_k … v_{i−1}.
//!
//! The search restricts v_i, v_j to the p nodes of the tour nearest to v,
//! v_k to the p nearest to v_{i+1}, and v_l to the p nearest to v_{j+1}, as
//! GENI does; with p ≥ L it is exhaustive.

/// One insertion, in the positions of the orientation `rev` names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geni {
    /// 0 plain, 1 type I, 2 type II.
    pub kind: u8,
    pub rev: bool,
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
    /// Length added to the tour.
    pub added: f64,
}

/// The tour in the orientation `rev` (t[0] stays first).
pub fn oriented(t: &[usize], rev: bool) -> Vec<usize> {
    if rev && t.len() > 1 {
        std::iter::once(t[0])
            .chain(t[1..].iter().rev().copied())
            .collect()
    } else {
        t.to_vec()
    }
}

/// Forward distance from position a to position b on a cycle of length n.
#[inline]
fn ahead(a: usize, b: usize, n: usize) -> usize {
    (b + n - a) % n
}

/// Whether (i, j, k) is a type I move on a tour of length n.
pub fn valid_1(n: usize, i: usize, j: usize, k: usize) -> bool {
    i < n && j < n && k < n && i != j && {
        let (jk, ji) = (ahead(j, k, n), ahead(j, i, n));
        0 < jk && jk < ji
    }
}

/// Whether (i, j, k, l) is a type II move on a tour of length n.
pub fn valid_2(n: usize, i: usize, j: usize, k: usize, l: usize) -> bool {
    i < n && j < n && k < n && l < n && i != j && {
        let (il, ij) = (ahead(i, l, n), ahead(i, j, n));
        // k = i is allowed (the last path is v_i alone): measure it as n
        let (jk, ji) = (ahead(j, k, n), ahead(j, i, n));
        2 <= il && il <= ij && 2 <= jk && jk <= ji
    }
}

/// Added length of a move on the oriented tour `t` (no validity check).
pub fn added(t: &[usize], v: usize, g: &Geni, d: &impl Fn(usize, usize) -> f64) -> f64 {
    let n = t.len();
    let at = |x: usize| t[x % n];
    let (i, j, k, l) = (g.i as usize, g.j as usize, g.k as usize, g.l as usize);
    match g.kind {
        0 => d(at(i), v) + d(v, at(i + 1)) - d(at(i), at(i + 1)),
        1 => {
            d(at(i), v) + d(v, at(j)) + d(at(i + 1), at(k)) + d(at(j + 1), at(k + 1))
                - d(at(i), at(i + 1))
                - d(at(j), at(j + 1))
                - d(at(k), at(k + 1))
        }
        _ => {
            let (lm, km) = (at(l + n - 1), at(k + n - 1));
            d(at(i), v) + d(v, at(j)) + d(at(l), at(j + 1)) + d(km, lm) + d(at(i + 1), at(k))
                - d(at(i), at(i + 1))
                - d(lm, at(l))
                - d(at(j), at(j + 1))
                - d(km, at(k))
        }
    }
}

/// The new cycle after the move, on the original (unoriented) tour `t`,
/// rotated so that t[0] is first.
pub fn apply(t: &[usize], v: usize, g: &Geni) -> Vec<usize> {
    let o = oriented(t, g.rev);
    let n = o.len();
    let at = |x: usize| o[x % n];
    let (i, j, k, l) = (g.i as usize, g.j as usize, g.k as usize, g.l as usize);
    let mut out = Vec::with_capacity(n + 1);
    // len nodes from position a, forward or backward
    let fwd = |out: &mut Vec<usize>, a: usize, len: usize| {
        for s in 0..len {
            out.push(at(a + s));
        }
    };
    let back = |out: &mut Vec<usize>, a: usize, len: usize| {
        for s in 0..len {
            out.push(at(a + n * len - s));
        }
    };
    match g.kind {
        0 => {
            fwd(&mut out, i + 1, n); // v_{i+1} … v_i
            out.push(v);
        }
        1 => {
            out.push(at(i));
            out.push(v);
            back(&mut out, j, ahead(i, j, n)); // v_j … v_{i+1}
            back(&mut out, k, ahead(j, k, n)); // v_k … v_{j+1}
            fwd(&mut out, k + 1, ahead(k, i, n) - 1); // v_{k+1} … v_{i−1}
        }
        _ => {
            out.push(at(i));
            out.push(v);
            back(&mut out, j, ahead(l, j, n) + 1); // v_j … v_l
            fwd(&mut out, j + 1, ahead(j, k, n) - 1); // v_{j+1} … v_{k−1}
            back(&mut out, l + n - 1, ahead(i, l, n) - 1); // v_{l−1} … v_{i+1}
            let ki = ahead(k, i, n); // v_k … v_{i−1}
            fwd(&mut out, k, ki);
        }
    }
    let first = out
        .iter()
        .position(|&x| x == t[0])
        .expect("t[0] on the tour");
    out.rotate_left(first);
    out
}

/// Every valid move on a tour of n nodes, both orientations (`added` unset).
pub fn moves(n: usize) -> Vec<Geni> {
    let g0 = Geni {
        kind: 0,
        rev: false,
        i: 0,
        j: 0,
        k: 0,
        l: 0,
        added: 0.0,
    };
    let mut out = Vec::new();
    for rev in [false, true] {
        for i in 0..n {
            let i32_ = i as u32;
            out.push(Geni { rev, i: i32_, ..g0 });
            for j in 0..n {
                for k in 0..n {
                    let g = Geni {
                        kind: 1,
                        rev,
                        i: i32_,
                        j: j as u32,
                        k: k as u32,
                        ..g0
                    };
                    if valid_1(n, i, j, k) {
                        out.push(g);
                    }
                    for l in 0..n {
                        if valid_2(n, i, j, k, l) {
                            out.push(Geni {
                                kind: 2,
                                l: l as u32,
                                ..g
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

/// Positions of the p nodes of `t` nearest to node x, skipping position `skip`.
fn nearest(
    t: &[usize],
    x: usize,
    skip: Option<usize>,
    p: usize,
    d: &impl Fn(usize, usize) -> f64,
    out: &mut Vec<usize>,
) {
    out.clear();
    out.extend((0..t.len()).filter(|&y| Some(y) != skip));
    if out.len() > p {
        out.select_nth_unstable_by(p - 1, |&a, &b| d(x, t[a]).total_cmp(&d(x, t[b])));
        out.truncate(p);
    }
}

/// GENI's p-neighbourhoods of every node of a tour, within the tour: row q
/// holds the positions (forward) of the min(p, L − 1) nodes nearest to t[q].
/// Built once per tour, shared by every vertex inserted into it.
pub struct Near {
    pub width: usize,
    pub rows: Vec<u32>,
}

impl Near {
    pub fn new(t: &[usize], p: usize, d: &impl Fn(usize, usize) -> f64) -> Self {
        let n = t.len();
        let width = p.min(n.saturating_sub(1));
        let mut rows = Vec::with_capacity(n * width);
        let mut buf = Vec::new();
        for q in 0..n {
            nearest(t, t[q], Some(q), width, d, &mut buf);
            rows.extend(buf.iter().map(|&x| x as u32));
        }
        Near { width, rows }
    }
}

/// The least-added-length insertion of v into tour t: plain insertion at
/// every position, and types I and II over GENI's p-neighbourhoods, in both
/// orientations.
pub fn best(t: &[usize], v: usize, p: usize, d: &impl Fn(usize, usize) -> f64) -> Geni {
    best_with(t, &Near::new(t, p, d), v, p, d)
}

/// `best` with the tour's neighbourhoods given (`near` built for this tour).
pub fn best_with(
    t: &[usize],
    near: &Near,
    v: usize,
    p: usize,
    d: &impl Fn(usize, usize) -> f64,
) -> Geni {
    let mut best = Geni {
        kind: 0,
        rev: false,
        i: 0,
        j: 0,
        k: 0,
        l: 0,
        added: f64::INFINITY,
    };
    let n = t.len();
    for i in 0..n {
        let (a, b) = (t[i], t[(i + 1) % n]);
        let add = d(a, v) + d(v, b) - d(a, b);
        if add < best.added {
            best = Geni {
                i: i as u32,
                added: add,
                ..best
            };
        }
    }
    if n < 3 || p == 0 {
        return best;
    }
    let w = near.width;
    let mut nv = Vec::new();
    for rev in [false, true] {
        // position in this orientation <-> forward position (an involution)
        let map = |q: usize| if rev { (n - q) % n } else { q };
        let at = |q: usize| t[map(q % n)];
        let row = |q: usize| {
            near.rows[map(q % n) * w..][..w]
                .iter()
                .map(move |&x| map(x as usize))
        };
        nearest(t, v, None, p, d, &mut nv);
        for x in nv.iter_mut() {
            *x = map(*x);
        }
        for &i in &nv {
            let (vi, vi1) = (at(i), at(i + 1));
            for &j in &nv {
                if i == j {
                    continue;
                }
                let (vj, vj1) = (at(j), at(j + 1));
                let base = d(vi, v) + d(v, vj) - d(vi, vi1) - d(vj, vj1);
                // type I: k in N(v_{i+1}) on v_{j+1} … v_{i−1}
                for k in row(i + 1) {
                    if valid_1(n, i, j, k) {
                        let (vk, vk1) = (at(k), at(k + 1));
                        let a = base + d(vi1, vk) + d(vj1, vk1) - d(vk, vk1);
                        if a < best.added {
                            best = Geni {
                                kind: 1,
                                rev,
                                i: i as u32,
                                j: j as u32,
                                k: k as u32,
                                l: 0,
                                added: a,
                            };
                        }
                    }
                }
                // type II: k in N(v_{i+1}), l in N(v_{j+1})
                for k in row(i + 1) {
                    let (vk, vkm) = (at(k), at(k + n - 1));
                    let bk = base + d(vi1, vk) - d(vkm, vk);
                    for l in row(j + 1) {
                        if valid_2(n, i, j, k, l) {
                            let (vl, vlm) = (at(l), at(l + n - 1));
                            let a = bk + d(vl, vj1) + d(vkm, vlm) - d(vlm, vl);
                            if a < best.added {
                                best = Geni {
                                    kind: 2,
                                    rev,
                                    i: i as u32,
                                    j: j as u32,
                                    k: k as u32,
                                    l: l as u32,
                                    added: a,
                                };
                            }
                        }
                    }
                }
            }
        }
    }
    best
}

/// Removal of a vertex v from a tour by unstringing (GENI's US), derived
/// as the inverse of the insertion types above. With o the tour in the
/// orientation `rev`, v at position p and o_x = o[(p + x) mod L] (offsets:
/// o_0 = v, a = o_{L−1} its predecessor):
///
/// * plain: a → o_1.
/// * type I (1 ≤ q < r ≤ L − 2), the inverse of insertion type I:
///   a, o_q … o_1, o_r … o_{q+1}, o_{r+1} … o_{L−2}; deletes (a, v),
///   (v, o_1), (o_q, o_{q+1}), (o_r, o_{r+1}); adds (a, o_q), (o_1, o_r),
///   (o_{q+1}, o_{r+1}).
/// * type II (1 ≤ q < r < s ≤ L − 2), the inverse of insertion type II:
///   a, o_s … o_{r+1}, o_q … o_1, o_{q+1} … o_r, o_{s+1} … o_{L−2}; deletes
///   (a, v), (v, o_1), (o_q, o_{q+1}), (o_r, o_{r+1}), (o_s, o_{s+1}); adds
///   (a, o_s), (o_{r+1}, o_q), (o_1, o_{q+1}), (o_r, o_{s+1}).
///
/// The search takes o_q (type I) and o_s (type II) among the p nearest to
/// a, o_r (type I) and o_{q+1} (type II) among the p nearest to o_1, and
/// o_r (type II) among the p nearest to o_{s+1}.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unstring {
    /// 0 plain, 1 type I, 2 type II.
    pub kind: u8,
    pub rev: bool,
    pub q: u32,
    pub r: u32,
    pub s: u32,
    /// Length change of the tour (usually negative).
    pub delta: f64,
}

/// Whether (q, r) is a type I unstringing on a tour of len nodes (v included).
pub fn valid_us1(len: usize, q: usize, r: usize) -> bool {
    1 <= q && q < r && r + 2 <= len
}

/// Whether (q, r, s) is a type II unstringing on a tour of len nodes.
pub fn valid_us2(len: usize, q: usize, r: usize, s: usize) -> bool {
    1 <= q && q < r && r < s && s + 2 <= len
}

/// Length change of an unstringing on the oriented tour o, v at position p.
pub fn unstring_delta(
    o: &[usize],
    p: usize,
    u: &Unstring,
    d: &impl Fn(usize, usize) -> f64,
) -> f64 {
    let n = o.len();
    let at = |x: usize| o[(p + x) % n];
    let (v, a, o1) = (at(0), at(n - 1), at(1));
    let (q, r, s) = (u.q as usize, u.r as usize, u.s as usize);
    let cut = d(a, v) + d(v, o1);
    match u.kind {
        0 => d(a, o1) - cut,
        1 => {
            d(a, at(q)) + d(o1, at(r)) + d(at(q + 1), at(r + 1))
                - cut
                - d(at(q), at(q + 1))
                - d(at(r), at(r + 1))
        }
        _ => {
            d(a, at(s)) + d(at(r + 1), at(q)) + d(o1, at(q + 1)) + d(at(r), at(s + 1))
                - cut
                - d(at(q), at(q + 1))
                - d(at(r), at(r + 1))
                - d(at(s), at(s + 1))
        }
    }
}

/// The cycle after removing v from t (v ≠ t[0]) by u, rotated so that
/// t[0] is first.
pub fn unstring(t: &[usize], v: usize, u: &Unstring) -> Vec<usize> {
    let o = oriented(t, u.rev);
    let n = o.len();
    let p = o.iter().position(|&x| x == v).expect("v on the tour");
    let at = |x: usize| o[(p + x) % n];
    let (q, r, s) = (u.q as usize, u.r as usize, u.s as usize);
    let mut out = vec![at(n - 1)];
    let span = |out: &mut Vec<usize>, from: usize, to: usize, back: bool| {
        // offsets from..=to, forward or backward (empty if from > to)
        if from <= to {
            if back {
                out.extend((from..=to).rev().map(at));
            } else {
                out.extend((from..=to).map(at));
            }
        }
    };
    match u.kind {
        0 => span(&mut out, 1, n - 2, false),
        1 => {
            span(&mut out, 1, q, true);
            span(&mut out, q + 1, r, true);
            span(&mut out, r + 1, n - 2, false);
        }
        _ => {
            span(&mut out, r + 1, s, true);
            span(&mut out, 1, q, true);
            span(&mut out, q + 1, r, false);
            span(&mut out, s + 1, n - 2, false);
        }
    }
    let first = out
        .iter()
        .position(|&x| x == t[0])
        .expect("t[0] on the tour");
    out.rotate_left(first);
    out
}

/// Every valid unstringing on a tour of len nodes, both orientations.
pub fn unstring_moves(len: usize) -> Vec<Unstring> {
    let u0 = Unstring {
        kind: 0,
        rev: false,
        q: 0,
        r: 0,
        s: 0,
        delta: 0.0,
    };
    let mut out = Vec::new();
    for rev in [false, true] {
        out.push(Unstring { rev, ..u0 });
        for q in 0..len {
            for r in 0..len {
                if valid_us1(len, q, r) {
                    out.push(Unstring {
                        kind: 1,
                        rev,
                        q: q as u32,
                        r: r as u32,
                        ..u0
                    });
                }
                for s in 0..len {
                    if valid_us2(len, q, r, s) {
                        out.push(Unstring {
                            kind: 2,
                            rev,
                            q: q as u32,
                            r: r as u32,
                            s: s as u32,
                            ..u0
                        });
                    }
                }
            }
        }
    }
    out
}

/// The best (least delta) removal of v from t (v ≠ t[0]): plain, and types I
/// and II over the p-neighbourhoods (`near` built for t, v included).
pub fn best_unstring(
    t: &[usize],
    near: &Near,
    v: usize,
    p: usize,
    d: &impl Fn(usize, usize) -> f64,
) -> Unstring {
    let n = t.len();
    let pos = t.iter().position(|&x| x == v).expect("v on the tour");
    let mut best = Unstring {
        kind: 0,
        rev: false,
        q: 0,
        r: 0,
        s: 0,
        delta: 0.0,
    };
    best.delta = unstring_delta(t, pos, &best, d);
    if n < 4 || p == 0 {
        return best;
    }
    let w = near.width;
    for rev in [false, true] {
        let map = |x: usize| if rev { (n - x) % n } else { x };
        let o = oriented(t, rev);
        let po = map(pos);
        // offsets from v of the nearest nodes to the node at offset x
        let off = |x: usize| {
            let f = map((po + x) % n);
            near.rows[f * w..][..w]
                .iter()
                .map(move |&y| (map(y as usize) + n - po) % n)
                .filter(|&y| y != 0)
        };
        let try_ = |best: &mut Unstring, u: Unstring| {
            let dl = unstring_delta(&o, po, &u, d);
            if dl < best.delta {
                *best = Unstring { delta: dl, ..u };
            }
        };
        let u0 = Unstring {
            kind: 1,
            rev,
            q: 0,
            r: 0,
            s: 0,
            delta: 0.0,
        };
        for q in off(n - 1) {
            for r in off(1) {
                if valid_us1(n, q, r) {
                    try_(
                        &mut best,
                        Unstring {
                            q: q as u32,
                            r: r as u32,
                            ..u0
                        },
                    );
                }
            }
        }
        for s in off(n - 1) {
            for q1 in off(1) {
                let q = q1.wrapping_sub(1);
                if q >= n {
                    continue;
                }
                for r in off(s + 1) {
                    if valid_us2(n, q, r, s) {
                        try_(
                            &mut best,
                            Unstring {
                                kind: 2,
                                q: q as u32,
                                r: r as u32,
                                s: s as u32,
                                ..u0
                            },
                        );
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use mets::rand::{Rng, SeedableRng};
    use mets::DefaultRng;
    use std::collections::HashMap;

    /// Random points 0..count in the plane and their Euclidean distance.
    fn points(count: usize, seed: u64) -> impl Fn(usize, usize) -> f64 {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let xy: Vec<(f64, f64)> = (0..count)
            .map(|_| (rng.random::<f64>() * 100.0, rng.random::<f64>() * 100.0))
            .collect();
        move |a, b| ((xy[a].0 - xy[b].0).powi(2) + (xy[a].1 - xy[b].1).powi(2)).sqrt()
    }

    fn length(c: &[usize], d: &impl Fn(usize, usize) -> f64) -> f64 {
        (0..c.len()).map(|x| d(c[x], c[(x + 1) % c.len()])).sum()
    }

    /// Undirected arcs of a cycle, with multiplicity (a 2-cycle has its arc twice).
    fn arcs(c: &[usize]) -> HashMap<(usize, usize), i32> {
        let mut m = HashMap::new();
        if c.len() < 2 {
            return m;
        }
        for x in 0..c.len() {
            let (a, b) = (c[x], c[(x + 1) % c.len()]);
            *m.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        m
    }

    /// The arcs the move should delete and add, straight from the GENI
    /// definitions (independent of `apply`).
    fn spec(o: &[usize], v: usize, g: &Geni) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let n = o.len();
        let at = |x: usize| o[x % n];
        let (i, j, k, l) = (g.i as usize, g.j as usize, g.k as usize, g.l as usize);
        match g.kind {
            0 => (vec![(at(i), at(i + 1))], vec![(at(i), v), (v, at(i + 1))]),
            1 => (
                vec![(at(i), at(i + 1)), (at(j), at(j + 1)), (at(k), at(k + 1))],
                vec![
                    (at(i), v),
                    (v, at(j)),
                    (at(i + 1), at(k)),
                    (at(j + 1), at(k + 1)),
                ],
            ),
            _ => (
                vec![
                    (at(i), at(i + 1)),
                    (at(l + n - 1), at(l)),
                    (at(j), at(j + 1)),
                    (at(k + n - 1), at(k)),
                ],
                vec![
                    (at(i), v),
                    (v, at(j)),
                    (at(l), at(j + 1)),
                    (at(k + n - 1), at(l + n - 1)),
                    (at(i + 1), at(k)),
                ],
            ),
        }
    }

    /// Check one valid move: a cycle through t and v with t[0] first, the
    /// arcs the definition says, and `added` equal to the length change.
    fn check(t: &[usize], v: usize, g: &Geni, d: &impl Fn(usize, usize) -> f64) {
        let c = apply(t, v, g);
        let mut sorted = c.clone();
        sorted.sort();
        let mut want: Vec<usize> = t.iter().copied().chain([v]).collect();
        want.sort();
        assert_eq!(
            sorted, want,
            "{g:?} on {t:?}: not a cycle through t and v: {c:?}"
        );
        assert_eq!(c[0], t[0], "{g:?}: t[0] first");
        let o = oriented(t, g.rev);
        let (del, add) = spec(&o, v, g);
        let mut expect = arcs(&o);
        if o.len() == 1 {
            // a one-node tour has no arc; plain insertion makes a 2-cycle
            assert_eq!(c.len(), 2);
        } else {
            for (a, b) in del {
                let e = expect.entry((a.min(b), a.max(b))).or_insert(0);
                *e -= 1;
                assert!(*e >= 0, "{g:?} on {o:?}: deletes ({a}, {b}), not an arc");
            }
            for (a, b) in add {
                *expect.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
            expect.retain(|_, &mut x| x != 0);
            assert_eq!(arcs(&c), expect, "{g:?} on {o:?}: arcs of {c:?}");
        }
        let delta = length(&c, d) - length(t, d);
        let a = added(&o, v, g, d);
        assert!(
            (a - delta).abs() < 1e-9,
            "{g:?} on {t:?}: added {a}, length change {delta}"
        );
    }

    /// Tours of 1 to 9 nodes: the depot alone (an empty route), then routes
    /// of 1, 2, 3, … customers. Every combination of positions is tried in
    /// both orientations, so every move with an index at the start (0), at
    /// the end (L − 1), next to another index, or wrapping round, is
    /// checked; each valid one must be a correct tour change.
    #[test]
    fn every_move_on_small_tours() {
        for len in 1..=9 {
            for seed in 0..3 {
                let d = points(len + 1, 10 * len as u64 + seed);
                let t: Vec<usize> = (0..len).collect();
                let v = len;
                let (mut n1, mut n2) = (0, 0);
                let mut ends = std::collections::HashSet::new();
                for g in moves(len) {
                    check(&t, v, &g, &d);
                    n1 += (g.kind == 1) as usize;
                    n2 += (g.kind == 2) as usize;
                    let last = len as u32 - 1;
                    for (name, x) in [("i", g.i), ("j", g.j), ("k", g.k), ("l", g.l)] {
                        if x == 0 || x == last {
                            ends.insert((g.kind, name, x == 0));
                        }
                    }
                }
                // type I needs three distinct positions, type II four segments
                assert_eq!(n1 > 0, len >= 3, "type I moves on {len} nodes");
                assert_eq!(n2 > 0, len >= 4, "type II moves on {len} nodes");
                if len >= 4 {
                    // the start and the end of the slice both occur for every index
                    for kind in [1u8, 2] {
                        let names: &[&str] = if kind == 1 {
                            &["i", "j", "k"]
                        } else {
                            &["i", "j", "k", "l"]
                        };
                        for &name in names {
                            for at_start in [true, false] {
                                assert!(
                                    ends.contains(&(kind, name, at_start)),
                                    "{len} nodes: no type {kind} move with {name} at the {}",
                                    if at_start { "start" } else { "end" }
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The validity rules accept exactly the moves the definitions allow:
    /// type I with k on the path v_{j+1} … v_{i−1}, type II with l on
    /// v_{i+2} … v_j and k on v_{j+2} … v_i.
    #[test]
    fn validity_rules_by_enumeration() {
        for n in 1..=8usize {
            for i in 0..n {
                for j in 0..n {
                    // the positions strictly after `from`, up to before `to`
                    let path = |from: usize, to: usize| -> Vec<usize> {
                        let mut p = Vec::new();
                        let mut x = (from + 1) % n;
                        while x != to {
                            p.push(x);
                            x = (x + 1) % n;
                        }
                        p
                    };
                    for k in 0..n {
                        let want = i != j && path(j, i).contains(&k);
                        assert_eq!(valid_1(n, i, j, k), want, "type I n={n} i={i} j={j} k={k}");
                        for l in 0..n {
                            // l in v_{i+2} … v_j, k in v_{j+2} … v_i
                            let ls = path((i + 1) % n, (j + 1) % n);
                            let ks = path((j + 1) % n, (i + 1) % n);
                            let want = i != j && ls.contains(&l) && ks.contains(&k);
                            assert_eq!(
                                valid_2(n, i, j, k, l),
                                want,
                                "type II n={n} i={i} j={j} k={k} l={l}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// With p at least the tour length, `best` is the exhaustive minimum.
    #[test]
    fn best_is_exhaustive_with_large_p() {
        for len in 1..=9 {
            for seed in 0..4 {
                let d = points(len + 1, 100 + 10 * len as u64 + seed);
                let t: Vec<usize> = (0..len).collect();
                let v = len;
                let brute = moves(len)
                    .into_iter()
                    .map(|g| added(&oriented(&t, g.rev), v, &g, &d))
                    .fold(f64::INFINITY, f64::min);
                let g = best(&t, v, len + 1, &d);
                assert!(
                    (g.added - brute).abs() < 1e-9,
                    "{len} nodes: best {} brute {brute}",
                    g.added
                );
                check(&t, v, &g, &d);
            }
        }
    }

    /// GENI never does worse than plain insertion, and with a small p on
    /// longer tours its chosen move is still a correct tour change.
    #[test]
    fn best_with_small_p_on_long_tours() {
        for len in [10, 25, 60] {
            for p in [1, 3, 5] {
                let d = points(len + 1, 7 * len as u64 + p as u64);
                let t: Vec<usize> = (0..len).collect();
                let g = best(&t, len, p, &d);
                let plain = (0..len)
                    .map(|i| d(t[i], len) + d(len, t[(i + 1) % len]) - d(t[i], t[(i + 1) % len]))
                    .fold(f64::INFINITY, f64::min);
                assert!(g.added <= plain + 1e-9);
                check(&t, len, &g, &d);
            }
        }
    }

    /// The arcs an unstringing should delete and add, from its definition.
    fn us_spec(o: &[usize], p: usize, u: &Unstring) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let n = o.len();
        let at = |x: usize| o[(p + x) % n];
        let (v, a, o1) = (at(0), at(n - 1), at(1));
        let (q, r, s) = (u.q as usize, u.r as usize, u.s as usize);
        let mut del = vec![(a, v), (v, o1)];
        let add = match u.kind {
            0 => vec![(a, o1)],
            1 => {
                del.extend([(at(q), at(q + 1)), (at(r), at(r + 1))]);
                vec![(a, at(q)), (o1, at(r)), (at(q + 1), at(r + 1))]
            }
            _ => {
                del.extend([(at(q), at(q + 1)), (at(r), at(r + 1)), (at(s), at(s + 1))]);
                vec![
                    (a, at(s)),
                    (at(r + 1), at(q)),
                    (o1, at(q + 1)),
                    (at(r), at(s + 1)),
                ]
            }
        };
        (del, add)
    }

    /// Check one valid unstringing: a cycle through t without v, t[0]
    /// first, the defined arcs, and `unstring_delta` equal to the length change.
    fn check_us(t: &[usize], v: usize, u: &Unstring, d: &impl Fn(usize, usize) -> f64) {
        let c = unstring(t, v, u);
        let mut sorted = c.clone();
        sorted.sort();
        let mut want: Vec<usize> = t.iter().copied().filter(|&x| x != v).collect();
        want.sort();
        assert_eq!(sorted, want, "{u:?} on {t:?} removing {v}: {c:?}");
        assert_eq!(c[0], t[0]);
        let o = oriented(t, u.rev);
        let p = o.iter().position(|&x| x == v).unwrap();
        let (del, add) = us_spec(&o, p, u);
        let mut expect = arcs(&o);
        for (a, b) in del {
            let e = expect.entry((a.min(b), a.max(b))).or_insert(0);
            *e -= 1;
            assert!(*e >= 0, "{u:?} on {o:?}: deletes ({a}, {b}), not an arc");
        }
        for (a, b) in add {
            *expect.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        expect.retain(|_, &mut x| x != 0);
        if c.len() >= 2 {
            assert_eq!(arcs(&c), expect, "{u:?} on {o:?}: arcs of {c:?}");
        }
        let delta = length(&c, d) - length(t, d);
        let dl = unstring_delta(&o, p, u, d);
        assert!(
            (dl - delta).abs() < 1e-9,
            "{u:?} on {t:?}: delta {dl}, length change {delta}"
        );
    }

    /// Tours of 2 to 10 nodes (a route of 1 to 9 customers, v among them),
    /// v at every position but the depot's, every valid unstringing in both
    /// orientations: every offset at its smallest and largest value, next
    /// to the others, and wrapping round past the depot.
    #[test]
    fn every_unstringing_on_small_tours() {
        for len in 2..=10 {
            for seed in 0..2 {
                let d = points(len, 300 + 10 * len as u64 + seed);
                let t: Vec<usize> = (0..len).collect();
                let moves = unstring_moves(len);
                let (n1, n2) = (
                    moves.iter().filter(|u| u.kind == 1).count(),
                    moves.iter().filter(|u| u.kind == 2).count(),
                );
                // type I needs v, a and two segments; type II three segments
                assert_eq!(n1 > 0, len >= 4, "type I unstringing on {len} nodes");
                assert_eq!(n2 > 0, len >= 5, "type II unstringing on {len} nodes");
                for u in &moves {
                    let (q, r, s) = (u.q as usize, u.r as usize, u.s as usize);
                    match u.kind {
                        1 => assert!(valid_us1(len, q, r)),
                        2 => assert!(valid_us2(len, q, r, s)),
                        _ => {}
                    }
                }
                if len >= 5 {
                    for (kind, get) in [(1u8, 0usize), (1, 1), (2, 0), (2, 1), (2, 2)] {
                        let vals: Vec<u32> = moves
                            .iter()
                            .filter(|u| u.kind == kind)
                            .map(|u| [u.q, u.r, u.s][get])
                            .collect();
                        let (lo, hi) = (*vals.iter().min().unwrap(), *vals.iter().max().unwrap());
                        // q from 1; r up to L − 2 for type I, s up to L − 2 for type II
                        assert_eq!(lo as usize, 1 + get, "type {kind} offset {get} smallest");
                        assert_eq!(
                            hi as usize,
                            len - 2 - if kind == 1 { 1 - get } else { 2 - get }
                        );
                    }
                }
                for v in 1..len {
                    for u in &moves {
                        check_us(&t, v, u, &d);
                    }
                }
            }
        }
    }

    /// Unstringing undoes insertion: for every insertion of v (types I and
    /// II) on tours of 3 to 8 nodes, an unstringing of the same type gives
    /// back the original cycle (as a set of arcs).
    #[test]
    fn unstringing_inverts_insertion() {
        for len in 3..=8 {
            let t: Vec<usize> = (0..len).collect();
            let v = len;
            let target = arcs(&t);
            for g in moves(len) {
                let c = apply(&t, v, &g);
                let undone = unstring_moves(c.len())
                    .into_iter()
                    .filter(|u| u.kind == g.kind)
                    .any(|u| arcs(&unstring(&c, v, &u)) == target);
                assert!(
                    undone,
                    "{g:?} on {len} nodes: no type {} unstringing restores it",
                    g.kind
                );
            }
        }
    }

    /// With p at least the tour length, `best_unstring` is the exhaustive
    /// minimum; with a small p its choice is still a correct removal.
    #[test]
    fn best_unstring_against_brute_force() {
        for len in 2..=10 {
            for seed in 0..3 {
                let d = points(len, 700 + 10 * len as u64 + seed);
                let t: Vec<usize> = (0..len).collect();
                for v in 1..len {
                    let brute = unstring_moves(len)
                        .iter()
                        .map(|u| {
                            let o = oriented(&t, u.rev);
                            let p = o.iter().position(|&x| x == v).unwrap();
                            unstring_delta(&o, p, u, &d)
                        })
                        .fold(f64::INFINITY, f64::min);
                    let u = best_unstring(&t, &Near::new(&t, len, &d), v, len, &d);
                    assert!((u.delta - brute).abs() < 1e-9, "{len} nodes, v = {v}");
                    check_us(&t, v, &u, &d);
                }
            }
        }
        for len in [12, 30, 60] {
            let d = points(len, 900 + len as u64);
            let t: Vec<usize> = (0..len).collect();
            for p in [1, 3, 5] {
                let near = Near::new(&t, p, &d);
                for v in [1, len / 2, len - 1] {
                    let u = best_unstring(&t, &near, v, p, &d);
                    check_us(&t, v, &u, &d);
                }
            }
        }
    }
}
