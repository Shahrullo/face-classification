def mappings():
    mapping_age = {
        0: '0-2 years',
        1: '3-9 years',
        2: '10-19 years',
        3: '20-29 years',
        4: '30-39 years',
        5: '40-49 years',
        6: '50-59 years',
        7: '60-69 years',
        8: '70 and above',
    }

    mapping_gender = {
        0: 'male',
        1: 'female'
    }

    mapping_race = {
        0: 'White',
        1: 'Black',
        2: 'Latino',
        3: 'East Asian',
        4: 'Sout East Asian',
        5: 'Indian',
        6: 'Middle Eastern'
    }

    return mapping_age, mapping_gender, mapping_race

