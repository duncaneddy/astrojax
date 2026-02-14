"""Tests for CelestrakQuery builder."""

import astrojax.celestrak as celestrak
from astrojax.spacetrack import operators as op


class TestCelestrakQueryConstructors:
    """Tests for CelestrakQuery class attribute constructors."""

    def test_gp_constructor(self):
        query = celestrak.CelestrakQuery.gp
        assert query.build_url() == ""
        assert "GP" in repr(query)

    def test_sup_gp_constructor(self):
        query = celestrak.CelestrakQuery.sup_gp
        assert query.build_url() == ""
        assert "SupGP" in repr(query)

    def test_satcat_constructor(self):
        query = celestrak.CelestrakQuery.satcat
        assert query.build_url() == ""
        assert "SATCAT" in repr(query)

    def test_fresh_instances(self):
        """Each access should return a fresh instance."""
        a = celestrak.CelestrakQuery.gp
        b = celestrak.CelestrakQuery.gp
        assert a is not b


class TestCelestrakQueryGP:
    """Tests for GP query builder methods."""

    def test_gp_by_group(self):
        query = celestrak.CelestrakQuery.gp.group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_gp_by_catnr(self):
        query = celestrak.CelestrakQuery.gp.catnr(25544)
        assert "CATNR=25544" in query.build_url()

    def test_gp_by_intdes(self):
        query = celestrak.CelestrakQuery.gp.intdes("1998-067A")
        assert "INTDES=1998-067A" in query.build_url()

    def test_gp_by_name(self):
        query = celestrak.CelestrakQuery.gp.name_search("ISS")
        assert "NAME=ISS" in query.build_url()

    def test_gp_with_format(self):
        query = celestrak.CelestrakQuery.gp.group("stations").format(
            celestrak.CelestrakOutputFormat.THREE_LE
        )
        assert "GROUP=stations" in query.build_url()
        assert "FORMAT=3LE" in query.build_url()

    def test_gp_with_special(self):
        query = celestrak.CelestrakQuery.gp.special("test")
        assert "SPECIAL=test" in query.build_url()


class TestCelestrakQuerySupGP:
    """Tests for SupGP query builder methods."""

    def test_sup_gp_by_source(self):
        query = celestrak.CelestrakQuery.sup_gp.source(
            celestrak.SupGPSource.SPACEX
        )
        assert "SOURCE=spacex" in query.build_url()

    def test_sup_gp_by_file(self):
        query = celestrak.CelestrakQuery.sup_gp.file("test.txt")
        assert "FILE=test.txt" in query.build_url()

    def test_sup_gp_by_catnr(self):
        query = celestrak.CelestrakQuery.sup_gp.catnr(25544)
        assert "CATNR=25544" in query.build_url()

    def test_sup_gp_with_format(self):
        query = celestrak.CelestrakQuery.sup_gp.source(
            celestrak.SupGPSource.STARLINK
        ).format(celestrak.CelestrakOutputFormat.JSON)
        url = query.build_url()
        assert "SOURCE=starlink" in url
        assert "FORMAT=JSON" in url


class TestCelestrakQuerySATCAT:
    """Tests for SATCAT query builder methods."""

    def test_satcat_active(self):
        query = celestrak.CelestrakQuery.satcat.active(True)
        assert "ACTIVE=Y" in query.build_url()

    def test_satcat_payloads(self):
        query = celestrak.CelestrakQuery.satcat.payloads(True)
        assert "PAYLOADS=Y" in query.build_url()

    def test_satcat_on_orbit(self):
        query = celestrak.CelestrakQuery.satcat.on_orbit(True)
        assert "ONORBIT=Y" in query.build_url()

    def test_satcat_max(self):
        query = celestrak.CelestrakQuery.satcat.max(100)
        assert "MAX=100" in query.build_url()

    def test_satcat_by_group(self):
        query = celestrak.CelestrakQuery.satcat.group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_satcat_by_name(self):
        query = celestrak.CelestrakQuery.satcat.name_search("ISS")
        assert "NAME=ISS" in query.build_url()

    def test_satcat_multiple_flags(self):
        query = (
            celestrak.CelestrakQuery.satcat.active(True)
            .payloads(True)
            .on_orbit(True)
        )
        url = query.build_url()
        assert "PAYLOADS=Y" in url
        assert "ONORBIT=Y" in url
        assert "ACTIVE=Y" in url

    def test_satcat_false_flags_not_in_url(self):
        query = celestrak.CelestrakQuery.satcat.active(False)
        assert "ACTIVE" not in query.build_url()


class TestCelestrakQueryClientSide:
    """Tests for client-side filter/order/limit methods."""

    def test_filter(self):
        query = celestrak.CelestrakQuery.gp.group("active").filter(
            "INCLINATION", ">50"
        )
        assert "GROUP=active" in query.build_url()
        assert query.has_client_side_processing()
        assert query.client_side_filters() == [("INCLINATION", ">50")]

    def test_order_by(self):
        query = celestrak.CelestrakQuery.gp.group("active").order_by(
            "INCLINATION", False
        )
        assert "GROUP=active" in query.build_url()
        assert query.has_client_side_processing()
        assert query.client_side_order_by() == [("INCLINATION", False)]

    def test_limit(self):
        query = celestrak.CelestrakQuery.gp.group("active").limit(10)
        assert "GROUP=active" in query.build_url()
        assert query.has_client_side_processing()
        assert query.client_side_limit() == 10

    def test_no_client_side_processing(self):
        query = celestrak.CelestrakQuery.gp.group("active")
        assert not query.has_client_side_processing()

    def test_filter_with_operators(self):
        """Test that SpaceTrack operators work as filter values."""
        query = (
            celestrak.CelestrakQuery.gp.group("active")
            .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
            .filter("INCLINATION", op.greater_than("50"))
            .filter("NORAD_CAT_ID", op.inclusive_range("25544", "25600"))
        )
        assert "GROUP=active" in query.build_url()
        filters = query.client_side_filters()
        assert len(filters) == 3
        assert filters[0] == ("OBJECT_TYPE", "<>DEBRIS")
        assert filters[1] == ("INCLINATION", ">50")
        assert filters[2] == ("NORAD_CAT_ID", "25544--25600")


class TestCelestrakQueryAccessors:
    """Tests for query accessor methods."""

    def test_query_type(self):
        assert celestrak.CelestrakQuery.gp.query_type() == celestrak.CelestrakQueryType.GP
        assert celestrak.CelestrakQuery.sup_gp.query_type() == celestrak.CelestrakQueryType.SUP_GP
        assert celestrak.CelestrakQuery.satcat.query_type() == celestrak.CelestrakQueryType.SATCAT

    def test_output_format_default(self):
        query = celestrak.CelestrakQuery.gp
        assert query.output_format() is None

    def test_output_format_set(self):
        query = celestrak.CelestrakQuery.gp.format(celestrak.CelestrakOutputFormat.JSON)
        assert query.output_format() == celestrak.CelestrakOutputFormat.JSON


class TestCelestrakQueryImmutability:
    """Test that builder methods return new instances."""

    def test_group_does_not_mutate(self):
        base = celestrak.CelestrakQuery.gp
        with_group = base.group("stations")
        assert base.build_url() == ""
        assert "GROUP=stations" in with_group.build_url()

    def test_filter_does_not_mutate(self):
        base = celestrak.CelestrakQuery.gp.group("active")
        with_filter = base.filter("INCLINATION", ">50")
        assert "GROUP=active" in base.build_url()
        assert "GROUP=active" in with_filter.build_url()
        assert not base.has_client_side_processing()
        assert with_filter.has_client_side_processing()

    def test_chaining(self):
        """Test fluent method chaining."""
        query = (
            celestrak.CelestrakQuery.gp.group("stations")
            .format(celestrak.CelestrakOutputFormat.JSON)
            .filter("INCLINATION", ">50")
            .order_by("INCLINATION", True)
            .limit(5)
        )
        url = query.build_url()
        assert "GROUP=stations" in url
        assert "FORMAT=JSON" in url

    def test_str(self):
        query = celestrak.CelestrakQuery.gp.group("stations")
        assert str(query) == query.build_url()

    def test_repr(self):
        query = celestrak.CelestrakQuery.gp.group("stations")
        assert "CelestrakQuery" in repr(query)
        assert "GP" in repr(query)
