def calculate_quarterly_payment(monthly_payments):
    if len(monthly_payments) != 3:
        return "Ошибка: Необходимо передать ровно 3 месячных платежа"

    quarterly_payment = sum(monthly_payments)
    return quarterly_payment


# Пример использования:
monthly_payments = [1000, 1200, 1100]
quarterly_payment = calculate_quarterly_payment(monthly_payments)
print("Квартальный платеж составляет:", quarterly_payment)
