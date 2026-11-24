"""Exact, invocation-local range sums for fixed-axis route segment prices."""
from bisect import bisect_left, bisect_right

from clay._native import load_native


class AxisPriceCache:
    """Memoize exact aligned segment sums without changing the partition.

    The caller owns one cache per immutable edge-search world. ``price`` is the
    original directed segment objective plus congestion. Every prefix segment,
    range lookup and partial-segment price is charged before it is evaluated.
    A refused charge returns None, retaining only fully computed prefixes.
    """
    def __init__(self, xs, ys, price, charge):
        self.axes = (tuple(xs), tuple(ys))
        if any(any(type(v) is not int for v in axis) or
               any(a >= b for a,b in zip(axis,axis[1:])) for axis in self.axes):
            raise ValueError('pricing axes must be strictly ordered integer coordinates')
        self.price, self.charge = price, charge
        self.prefixes = {}
        self._range_totals = {}
        cpp = load_native("_gridless_cpp")
        if cpp is not None:
            try:
                self._native = cpp.NativeAxisPriceCache(self.axes[0], self.axes[1])
            except OverflowError:
                self._native = None
        else:
            self._native = None

    def _range(self, gi, horizontal, fixed, sign, begin, end):
        axes = self.axes[0 if horizontal else 1]
        line = (gi, horizontal, fixed, sign)
        if begin >= end:
            return 0
        range_key = (line, begin, end)
        cached = self._range_totals.get(range_key)
        if cached is not None:
            return cached if self.charge('price_range') else None
        segs = self.prefixes.get(line)
        if segs is None:
            # Lazily memoize only queried elementary intervals on this line so
            # every unique segment is evaluated and charged at most once.
            segs = [None] * (len(axes) - 1)
            self.prefixes[line] = segs
        total = 0
        for i in range(begin, end):
            value = segs[i]
            if value is None:
                if not self.charge('price_segment'):
                    return None
                a, b = axes[i], axes[i + 1]
                if sign < 0:
                    a, b = b, a
                value = self.price(gi, a, fixed, b, fixed) if horizontal else self.price(gi, fixed, a, fixed, b)
                segs[i] = value
            total += value
        self._range_totals[range_key] = total
        if not self.charge('price_range'):
            return None
        return total

    def run(self, gi, start, end):
        native = self._native
        if native is not None:
            return native.run(gi, start, end, self.price, self.charge)
        ax,ay=start; bx,by=end
        if start == end:
            return 0
        if ax != bx and ay != by:
            raise ValueError('cost run must be orthogonal')
        horizontal = ay == by
        first_coord, last_coord = (ax,bx) if horizontal else (ay,by)
        fixed = ay if horizontal else ax
        sign = 1 if last_coord > first_coord else -1
        axes = self.axes[0 if horizontal else 1]
        lo,hi = sorted((first_coord,last_coord))
        first,stop = bisect_right(axes,lo),bisect_left(axes,hi)
        def partial(a,b):
            if not self.charge('price_segment'):
                return None
            return self.price(gi,a,fixed,b,fixed) if horizontal else self.price(gi,fixed,a,fixed,b)
        if first == stop:
            return partial(first_coord,last_coord)
        leading = axes[first] if sign > 0 else axes[stop-1]
        trailing = axes[stop-1] if sign > 0 else axes[first]
        a = partial(first_coord,leading)
        if a is None:
            return None
        middle = self._range(gi,horizontal,fixed,sign,first,stop-1)
        if middle is None:
            return None
        b = partial(trailing,last_coord)
        return None if b is None else a+middle+b

    def run_ctx(
        self,
        gi,
        start,
        end,
        edge_ctx,
        remaining_budget,
        static_price_steps,
        price_cache_hits,
        max_run_cache,
        watchdog_cb,
        fallback_price_fn,
    ):
        """Evaluate cached orthogonal range price natively via NativeEdgeContext."""
        return self._native.run_ctx(
            gi,
            start,
            end,
            edge_ctx,
            remaining_budget,
            static_price_steps,
            price_cache_hits,
            max_run_cache,
            watchdog_cb,
            fallback_price_fn,
        )
